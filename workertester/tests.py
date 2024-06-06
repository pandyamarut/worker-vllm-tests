# imports for guided decoding tests
import json
import re

import jsonschema
import openai  # use the official client for correctness check
import pytest
pytestmark = pytest.mark.asyncio
# using Ray for overall ease of process management, parallel requests,
# and debugging.
import ray
import torch
# downloading lora to test lora requests
from huggingface_hub import snapshot_download
from openai import BadRequestError


TEST_SCHEMA = {
	"type": "object",
	"properties": {
		"name": {
			"type": "string"
		},
		"age": {
			"type": "integer"
		},
		"skills": {
			"type": "array",
			"items": {
				"type": "string",
				"maxLength": 10
			},
			"minItems": 3
		},
		"work history": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"company": {
						"type": "string"
					},
					"duration": {
						"type": "string"
					},
					"position": {
						"type": "string"
					}
				},
				"required": ["company", "position"]
			}
		}
	},
	"required": ["name", "age", "skills", "work history"]
}

TEST_REGEX = (r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
			  r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)")

TEST_CHOICE = [
	"Python", "Java", "JavaScript", "C++", "C#", "PHP", "TypeScript", "Ruby",
	"Swift", "Kotlin"
]


# any model with a chat template should work here
class OpenAITest:
	def __init__(self, model_name, base_url, api_key):
		self.api_key = api_key
		self.model_name = model_name
		self.client = openai.AsyncOpenAI(
			base_url=base_url,
			api_key=api_key,
		)
		
	async def test_check_models(self):
		"""
		Check that /models endpoint returns the expected model
		"""
		models = await self.client.models.list()
		models = models.data
		served_model = models[0]
		assert served_model.id == self.model_name
		
	async def test_single_completion(self):
		completion = await self.client.completions.create(model=self.model_name,
													prompt="Hello, my name is",
													max_tokens=5,
													temperature=0.0)

		assert completion.id is not None
		assert completion.choices is not None and len(completion.choices) == 1
		assert completion.choices[0].text is not None and len(
			completion.choices[0].text) >= 5
		assert completion.choices[0].finish_reason == "length"
		assert completion.usage == openai.types.CompletionUsage(
			completion_tokens=5, prompt_tokens=6, total_tokens=11)

		# test using token IDs
		completion = await self.client.completions.create(
			model=self.model_name,
			prompt=[0, 0, 0, 0, 0],
			max_tokens=5,
			temperature=0.0,
		)
		assert completion.choices[0].text is not None and len(
			completion.choices[0].text) >= 5


	async def test_single_chat_session(self):
		"""
		Check that a single chat session works
		"""
		
		messages = [{
			"role": "system",
			"content": "you are a helpful assistant"
		}, {
			"role": "user",
			"content": "what is 1+1?"
		}]

		# test single completion
		chat_completion = await self.client.chat.completions.create(model=self.model_name,
															messages=messages,
															max_tokens=10,
															logprobs=True,
															top_logprobs=5)
		assert chat_completion.id is not None
		assert chat_completion.choices is not None and len(
			chat_completion.choices) == 1
		assert chat_completion.choices[0].message is not None
		assert chat_completion.choices[0].logprobs is not None
		assert chat_completion.choices[0].logprobs.top_logprobs is not None
		assert len(chat_completion.choices[0].logprobs.top_logprobs[0]) == 5
		message = chat_completion.choices[0].message
		assert message.content is not None and len(message.content) >= 10
		assert message.role == "assistant"
		messages.append({"role": "assistant", "content": message.content})

		# test multi-turn dialogue
		messages.append({"role": "user", "content": "express your result in json"})
		chat_completion = await self.client.chat.completions.create(
			model=self.model_name,
			messages=messages,
			max_tokens=10,
		)
		message = chat_completion.choices[0].message
		assert message.content is not None and len(message.content) >= 0

	async def test_completion_streaming(self):
		prompt = "What is an LLM?"

		single_completion = await self.client.completions.create(
			model=self.model_name,
			prompt=prompt,
			max_tokens=5,
			temperature=0.0,
		)
		single_output = single_completion.choices[0].text
		single_usage = single_completion.usage

		stream = await self.client.completions.create(model=self.model_name,
												prompt=prompt,
												max_tokens=5,
												temperature=0.0,
												stream=True)
		chunks = []
		finish_reason_count = 0
		async for chunk in stream:
			chunks.append(chunk.choices[0].text)
			if chunk.choices[0].finish_reason is not None:
				finish_reason_count += 1
		# finish reason should only return in last block
		assert finish_reason_count == 1
		assert chunk.choices[0].finish_reason == "length"
		assert chunk.choices[0].text
		assert chunk.usage == single_usage
		assert "".join(chunks) == single_output
			
	async def test_chat_streaming(self):
		messages = [{
			"role": "system",
			"content": "you are a helpful assistant"
		}, {
			"role": "user",
			"content": "what is 1+1?"
		}]

		# test single completion
		chat_completion = await self.client.chat.completions.create(
			model=self.model_name,
			messages=messages,
			max_tokens=10,
			temperature=0.0,
		)
		output = chat_completion.choices[0].message.content
		stop_reason = chat_completion.choices[0].finish_reason

		# test streaming
		stream = await self.client.chat.completions.create(
			model=self.model_name,
			messages=messages,
			max_tokens=10,
			temperature=0.0,
			stream=True,
		)
		chunks = []
		finish_reason_count = 0
		async for chunk in stream:
			delta = chunk.choices[0].delta
			if delta.role:
				assert delta.role == "assistant"
			if delta.content:
				chunks.append(delta.content)
			if chunk.choices[0].finish_reason is not None:
				finish_reason_count += 1
		# finish reason should only return in last block
		assert finish_reason_count == 1
		assert chunk.choices[0].finish_reason == stop_reason
		assert delta.content
		assert "".join(chunks) == output


	async def test_complex_message_content(self):
		resp = await self.client.chat.completions.create(
			model=self.model_name,
			messages=[{
				"role":
				"user",
				"content": [{
					"type":
					"text",
					"text":
					"what is 1+1? please provide the result without any other text."
				}]
			}],
			temperature=0,
			seed=0)
		content = resp.choices[0].message.content
		assert content == "2"

	async def test_extra_fields(self):
		try:
			await self.client.chat.completions.create(
				model=self.model_name,
				messages=[{
					"role": "system",
					"content": "You are a helpful assistant.",
					"extra_field": "0",
				}],  # type: ignore
				temperature=0,
				seed=0)
		except BadRequestError as e:
			assert "extra_forbidden" in e.message

	async def test_guided_choice_completion(self, guided_decoding_backend: str):
		completion = await self.client.completions.create(
			model=self.model_name,
			prompt="The best language for type-safe systems programming is ",
			n=2,
			temperature=1.0,
			max_tokens=10,
			extra_body=dict(guided_choice=TEST_CHOICE,
						guided_decoding_backend=guided_decoding_backend))

		assert completion.id is not None
		assert completion.choices is not None and len(completion.choices) == 2
		for i in range(2):
			assert completion.choices[i].text in TEST_CHOICE
   
	async def test_completion_streaming(self):
		prompt = "What is an LLM?"

		single_completion = await self.client.completions.create(
			model=self.model_name,
			prompt=prompt,
			max_tokens=5,
			temperature=0.0,
		)
		single_output = single_completion.choices[0].text
		single_usage = single_completion.usage

		stream = await self.client.completions.create(model=self.model_name,
												 prompt=prompt,
												 max_tokens=5,
												 temperature=0.0,
												 stream=True)
		chunks = []
		finish_reason_count = 0
		async for chunk in stream:
			chunks.append(chunk.choices[0].text)
			if chunk.choices[0].finish_reason is not None:
				finish_reason_count += 1
		# finish reason should only return in last block
		assert finish_reason_count == 1
		assert chunk.choices[0].finish_reason == "length"
		assert chunk.choices[0].text
		assert chunk.usage == single_usage
		assert "".join(chunks) == single_output
    
	async def test_batch_completions(self):
		# test simple list
		batch = await self.client.completions.create(
			model=self.model_name,
			prompt=["Hello, my name is", "Hello, my name is"],
			max_tokens=5,
			temperature=0.0,
		)
		assert len(batch.choices) == 2
		assert batch.choices[0].text == batch.choices[1].text

		# test n = 2
		batch = await self.client.completions.create(
			model=self.model_name,
			prompt=["Hello, my name is", "Hello, my name is"],
			n=2,
			max_tokens=5,
			temperature=0.0,
			extra_body=dict(
				# NOTE: this has to be true for n > 1 in vLLM, but not necessary
				# for official client.
				use_beam_search=True),
		)
		assert len(batch.choices) == 4
		assert batch.choices[0].text != batch.choices[1].text, "beam search should be different"
		assert batch.choices[0].text == batch.choices[2].text, "two copies of the same prompt should be the same"
		assert batch.choices[1].text == batch.choices[3].text, "two copies of the same prompt should be the same"

		# test streaming
		batch = await self.client.completions.create(
			model=self.model_name,
			prompt=["Hello, my name is", "Hello, my name is"],
			max_tokens=5,
			temperature=0.0,
			stream=True,
		)
		texts = [""] * 2
		async for chunk in batch:
			assert len(chunk.choices) == 1
			choice = chunk.choices[0]
			texts[choice.index] += choice.text
		assert texts[0] == texts[1]
    
    
    
	# async def test_long_seed(self):
	# 	for seed in [
	# 			torch.iinfo(torch.long).min - 1,
	# 			torch.iinfo(torch.long).max + 1
	# 	]:
	# 		with pytest.raises(BadRequestError) as exc_info:
	# 			await self.client.chat.completions.create(
	# 				model=self.model_name,
	# 				messages=[{
	# 					"role": "system",
	# 					"content": "You are a helpful assistant.",
	# 				}],
	# 				temperature=0,
	# 				seed=seed
	# 			)
	# 		assert ("greater_than_equal" in exc_info.value.message
	# 				or "less_than_equal" in exc_info.value.message)


	async def test_guided_decoding_type_error(self):
		guided_decoding_backends = ["outlines", "lm-format-enforcer"]

		for guided_decoding_backend in guided_decoding_backends:
			try:
				await self.client.completions.create(
					model=self.model_name,
					prompt="Give an example JSON that fits this schema: 42",
					extra_body=dict(guided_json=42, guided_decoding_backend=guided_decoding_backend)
				)
			except BadRequestError:
				pass
			else:
				raise AssertionError("BadRequestError was not raised for guided_decoding_backend: " + guided_decoding_backend)

		messages = [
			{"role": "system", "content": "you are a helpful assistant"},
			{"role": "user", "content": "The best language for type-safe systems programming is "}
		]

		try:
			await self.client.chat.completions.create(
				model=self.model_name,
				messages=messages,
				extra_body=dict(guided_regex={1: "Python", 2: "C++"})
			)
		except BadRequestError:
			pass
		else:
			raise AssertionError("BadRequestError was not raised for guided_regex with messages")

		try:
			await self.client.completions.create(
				model=self.model_name,			prompt="Give an example string that fits this regex",
				extra_body=dict(guided_regex=TEST_REGEX, guided_json=TEST_SCHEMA)
			)
		except BadRequestError:
			pass
		else:
			raise AssertionError("BadRequestError was not raised for guided_regex with prompt and schema")
 
	async def test_custom_role(self):
		# Not sure how the model handles custom roles so we just check that
		# both string and complex message content are handled in the same way

		resp1 = await self.client.chat.completions.create(
			model=self.model_name,
			messages=[{
				"role": "my-custom-role",
				"content": "what is 1+1?",
			}],  # type: ignore
			temperature=0,
			seed=0)

		resp2 = await self.client.chat.completions.create(
			model=self.model_name,
			messages=[{
				"role": "my-custom-role",
				"content": [{
					"type": "text",
					"text": "what is 1+1?"
				}]
			}],  # type: ignore
			temperature=0,
			seed=0)

		content1 = resp1.choices[0].message.content
		content2 = resp2.choices[0].message.content
		assert content1 == content2
 
	async def test_zero_logprobs(self):
		# test using token IDs
		completion = await self.client.completions.create(
			model=self.model_name,
			prompt=[0, 0, 0, 0, 0],
			max_tokens=5,
			temperature=0.0,
			logprobs=0,
		)
		choice = completion.choices[0]
		assert choice.logprobs is not None
		assert choice.logprobs.token_logprobs is not None
		assert choice.logprobs.top_logprobs is None
 
	async def run_tests(self):
		results = {}
		for name, obj in self.__class__.__dict__.items():
			if name.startswith("test_") and callable(obj):
				try:
					await obj(self)
					results[name] ={"status": "pass", "error": None}
				except Exception as e:
					results[name] = {"status": "fail", "error": str(e)}
		return results
	
 
	