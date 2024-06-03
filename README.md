
FUTURE TESTS
```python

# @pytest.mark.parametrize(
#     # first test base model, then test loras
#     "model_name",
#     [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
# )
# async def test_single_completion(server, client: openai.AsyncOpenAI,
#                                  model_name: str):
#     completion = await client.completions.create(model=model_name,
#                                                  prompt="Hello, my name is",
#                                                  max_tokens=5,
#                                                  temperature=0.0)

#     assert completion.id is not None
#     assert completion.choices is not None and len(completion.choices) == 1
#     assert completion.choices[0].text is not None and len(
#         completion.choices[0].text) >= 5
#     assert completion.choices[0].finish_reason == "length"
#     assert completion.usage == openai.types.CompletionUsage(
#         completion_tokens=5, prompt_tokens=6, total_tokens=11)

#     # test using token IDs
#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt=[0, 0, 0, 0, 0],
#         max_tokens=5,
#         temperature=0.0,
#     )
#     assert completion.choices[0].text is not None and len(
#         completion.choices[0].text) >= 5


# @pytest.mark.parametrize(
#     # first test base model, then test loras
#     "model_name",
#     [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
# )
# async def test_zero_logprobs(server, client: openai.AsyncOpenAI,
#                              model_name: str):
#     # test using token IDs
#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt=[0, 0, 0, 0, 0],
#         max_tokens=5,
#         temperature=0.0,
#         logprobs=0,
#     )
#     choice = completion.choices[0]
#     assert choice.logprobs is not None
#     assert choice.logprobs.token_logprobs is not None
#     assert choice.logprobs.top_logprobs is None


# @pytest.mark.parametrize("model_name", [MODEL_NAME])
# async def test_too_many_logprobs(server, client: openai.AsyncOpenAI,
#                                  model_name: str):
#     messages = [{
#         "role": "system",
#         "content": "you are a helpful assistant"
#     }, {
#         "role": "user",
#         "content": "what is 1+1?"
#     }]

#     # Default max_logprobs is 5, so this should raise an error
#     with pytest.raises((openai.BadRequestError, openai.APIError)):
#         stream = await client.chat.completions.create(model=model_name,
#                                                       messages=messages,
#                                                       max_tokens=10,
#                                                       logprobs=True,
#                                                       top_logprobs=10,
#                                                       stream=True)
#         async for chunk in stream:
#             ...

#     with pytest.raises(openai.BadRequestError):
#         await client.chat.completions.create(model=model_name,
#                                              messages=messages,
#                                              max_tokens=10,
#                                              logprobs=True,
#                                              top_logprobs=10,
#                                              stream=False)

#     with pytest.raises((openai.BadRequestError, openai.APIError)):
#         stream = await client.completions.create(model=model_name,
#                                                  prompt="Test",
#                                                  max_tokens=10,
#                                                  logprobs=10,
#                                                  stream=True)
#         async for chunk in stream:
#             ...

#     with pytest.raises(openai.BadRequestError):
#         await client.completions.create(model=model_name,
#                                         prompt="Test",
#                                         max_tokens=10,
#                                         logprobs=10,
#                                         stream=False)

#     # the server should still work afterwards
#     chat_completion = await client.chat.completions.create(model=model_name,
#                                                            messages=messages,
#                                                            max_tokens=10,
#                                                            stream=False)
#     message = chat_completion.choices[0].message
#     assert message.content is not None and len(message.content) >= 0

# @pytest.mark.parametrize(
#     # just test 1 lora hereafter
#     "model_name",
#     [MODEL_NAME, "zephyr-lora"],
# )
# async def test_batch_completions(server, client: openai.AsyncOpenAI,
#                                  model_name: str):
#     # test simple list
#     batch = await client.completions.create(
#         model=model_name,
#         prompt=["Hello, my name is", "Hello, my name is"],
#         max_tokens=5,
#         temperature=0.0,
#     )
#     assert len(batch.choices) == 2
#     assert batch.choices[0].text == batch.choices[1].text

#     # test n = 2
#     batch = await client.completions.create(
#         model=model_name,
#         prompt=["Hello, my name is", "Hello, my name is"],
#         n=2,
#         max_tokens=5,
#         temperature=0.0,
#         extra_body=dict(
#             # NOTE: this has to be true for n > 1 in vLLM, but not necessary
#             # for official client.
#             use_beam_search=True),
#     )
#     assert len(batch.choices) == 4
#     assert batch.choices[0].text != batch.choices[
#         1].text, "beam search should be different"
#     assert batch.choices[0].text == batch.choices[
#         2].text, "two copies of the same prompt should be the same"
#     assert batch.choices[1].text == batch.choices[
#         3].text, "two copies of the same prompt should be the same"

#     # test streaming
#     batch = await client.completions.create(
#         model=model_name,
#         prompt=["Hello, my name is", "Hello, my name is"],
#         max_tokens=5,
#         temperature=0.0,
#         stream=True,
#     )
#     texts = [""] * 2
#     async for chunk in batch:
#         assert len(chunk.choices) == 1
#         choice = chunk.choices[0]
#         texts[choice.index] += choice.text
#     assert texts[0] == texts[1]


# async def test_logits_bias(server, client: openai.AsyncOpenAI):
#     prompt = "Hello, my name is"
#     max_tokens = 5
#     tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)

#     # Test exclusive selection
#     token_id = 1000
#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt=prompt,
#         max_tokens=max_tokens,
#         temperature=0.0,
#         logit_bias={str(token_id): 100},
#         seed=42,
#     )
#     assert completion.choices[0].text is not None and len(
#         completion.choices[0].text) >= 5
#     response_tokens = tokenizer(completion.choices[0].text,
#                                 add_special_tokens=False)["input_ids"]
#     expected_tokens = tokenizer(tokenizer.decode([token_id] * 5),
#                                 add_special_tokens=False)["input_ids"]
#     assert all([
#         response == expected
#         for response, expected in zip(response_tokens, expected_tokens)
#     ])

#     # Test ban
#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt=prompt,
#         max_tokens=max_tokens,
#         temperature=0.0,
#     )
#     response_tokens = tokenizer(completion.choices[0].text,
#                                 add_special_tokens=False)["input_ids"]
#     first_response = completion.choices[0].text
#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt=prompt,
#         max_tokens=max_tokens,
#         temperature=0.0,
#         logit_bias={str(token): -100
#                     for token in response_tokens},
#     )
#     assert first_response != completion.choices[0].text


# @pytest.mark.parametrize("guided_decoding_backend",
#                          ["outlines", "lm-format-enforcer"])
# async def test_guided_json_completion(server, client: openai.AsyncOpenAI,
#                                       guided_decoding_backend: str):
#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt=f"Give an example JSON for an employee profile "
#         f"that fits this schema: {TEST_SCHEMA}",
#         n=3,
#         temperature=1.0,
#         max_tokens=500,
#         extra_body=dict(guided_json=TEST_SCHEMA,
#                         guided_decoding_backend=guided_decoding_backend))

#     assert completion.id is not None
#     assert completion.choices is not None and len(completion.choices) == 3
#     for i in range(3):
#         assert completion.choices[i].text is not None
#         output_json = json.loads(completion.choices[i].text)
#         jsonschema.validate(instance=output_json, schema=TEST_SCHEMA)


# @pytest.mark.parametrize("guided_decoding_backend",
#                          ["outlines", "lm-format-enforcer"])
# async def test_guided_json_chat(server, client: openai.AsyncOpenAI,
#                                 guided_decoding_backend: str):
#     messages = [{
#         "role": "system",
#         "content": "you are a helpful assistant"
#     }, {
#         "role":
#         "user",
#         "content":
#         f"Give an example JSON for an employee profile that "
#         f"fits this schema: {TEST_SCHEMA}"
#     }]
#     chat_completion = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         max_tokens=1000,
#         extra_body=dict(guided_json=TEST_SCHEMA,
#                         guided_decoding_backend=guided_decoding_backend))
#     message = chat_completion.choices[0].message
#     assert message.content is not None
#     json1 = json.loads(message.content)
#     jsonschema.validate(instance=json1, schema=TEST_SCHEMA)

#     messages.append({"role": "assistant", "content": message.content})
#     messages.append({
#         "role":
#         "user",
#         "content":
#         "Give me another one with a different name and age"
#     })
#     chat_completion = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         max_tokens=1000,
#         extra_body=dict(guided_json=TEST_SCHEMA,
#                         guided_decoding_backend=guided_decoding_backend))
#     message = chat_completion.choices[0].message
#     assert message.content is not None
#     json2 = json.loads(message.content)
#     jsonschema.validate(instance=json2, schema=TEST_SCHEMA)
#     assert json1["name"] != json2["name"]
#     assert json1["age"] != json2["age"]


# @pytest.mark.parametrize("guided_decoding_backend",
#                          ["outlines", "lm-format-enforcer"])
# async def test_guided_regex_completion(server, client: openai.AsyncOpenAI,
#                                        guided_decoding_backend: str):
#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt=f"Give an example IPv4 address with this regex: {TEST_REGEX}",
#         n=3,
#         temperature=1.0,
#         max_tokens=20,
#         extra_body=dict(guided_regex=TEST_REGEX,
#                         guided_decoding_backend=guided_decoding_backend))

#     assert completion.id is not None
#     assert completion.choices is not None and len(completion.choices) == 3
#     for i in range(3):
#         assert completion.choices[i].text is not None
#         assert re.fullmatch(TEST_REGEX, completion.choices[i].text) is not None


# @pytest.mark.parametrize("guided_decoding_backend",
#                          ["outlines", "lm-format-enforcer"])
# async def test_guided_regex_chat(server, client: openai.AsyncOpenAI,
#                                  guided_decoding_backend: str):
#     messages = [{
#         "role": "system",
#         "content": "you are a helpful assistant"
#     }, {
#         "role":
#         "user",
#         "content":
#         f"Give an example IP address with this regex: {TEST_REGEX}"
#     }]
#     chat_completion = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         max_tokens=20,
#         extra_body=dict(guided_regex=TEST_REGEX,
#                         guided_decoding_backend=guided_decoding_backend))
#     ip1 = chat_completion.choices[0].message.content
#     assert ip1 is not None
#     assert re.fullmatch(TEST_REGEX, ip1) is not None

#     messages.append({"role": "assistant", "content": ip1})
#     messages.append({"role": "user", "content": "Give me a different one"})
#     chat_completion = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         max_tokens=20,
#         extra_body=dict(guided_regex=TEST_REGEX,
#                         guided_decoding_backend=guided_decoding_backend))
#     ip2 = chat_completion.choices[0].message.content
#     assert ip2 is not None
#     assert re.fullmatch(TEST_REGEX, ip2) is not None
#     assert ip1 != ip2


# @pytest.mark.parametrize("guided_decoding_backend",
#                          ["outlines", "lm-format-enforcer"])
# async def test_guided_choice_completion(server, client: openai.AsyncOpenAI,
#                                         guided_decoding_backend: str):
#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt="The best language for type-safe systems programming is ",
#         n=2,
#         temperature=1.0,
#         max_tokens=10,
#         extra_body=dict(guided_choice=TEST_CHOICE,
#                         guided_decoding_backend=guided_decoding_backend))

#     assert completion.id is not None
#     assert completion.choices is not None and len(completion.choices) == 2
#     for i in range(2):
#         assert completion.choices[i].text in TEST_CHOICE


# @pytest.mark.parametrize("guided_decoding_backend",
#                          ["outlines", "lm-format-enforcer"])
# async def test_guided_choice_chat(server, client: openai.AsyncOpenAI,
#                                   guided_decoding_backend: str):
#     messages = [{
#         "role": "system",
#         "content": "you are a helpful assistant"
#     }, {
#         "role":
#         "user",
#         "content":
#         "The best language for type-safe systems programming is "
#     }]
#     chat_completion = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         max_tokens=10,
#         extra_body=dict(guided_choice=TEST_CHOICE,
#                         guided_decoding_backend=guided_decoding_backend))
#     choice1 = chat_completion.choices[0].message.content
#     assert choice1 in TEST_CHOICE

#     messages.append({"role": "assistant", "content": choice1})
#     messages.append({
#         "role": "user",
#         "content": "I disagree, pick another one"
#     })
#     chat_completion = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         max_tokens=10,
#         extra_body=dict(guided_choice=TEST_CHOICE,
#                         guided_decoding_backend=guided_decoding_backend))
#     choice2 = chat_completion.choices[0].message.content
#     assert choice2 in TEST_CHOICE
#     assert choice1 != choice2


# @pytest.mark.parametrize("guided_decoding_backend",
#                          ["outlines", "lm-format-enforcer"])
# async def test_guided_decoding_type_error(server, client: openai.AsyncOpenAI,
#                                           guided_decoding_backend: str):
#     with pytest.raises(openai.BadRequestError):
#         _ = await client.completions.create(
#             model=MODEL_NAME,
#             prompt="Give an example JSON that fits this schema: 42",
#             extra_body=dict(guided_json=42,
#                             guided_decoding_backend=guided_decoding_backend))

#     messages = [{
#         "role": "system",
#         "content": "you are a helpful assistant"
#     }, {
#         "role":
#         "user",
#         "content":
#         "The best language for type-safe systems programming is "
#     }]
#     with pytest.raises(openai.BadRequestError):
#         _ = await client.chat.completions.create(model=MODEL_NAME,
#                                                  messages=messages,
#                                                  extra_body=dict(guided_regex={
#                                                      1: "Python",
#                                                      2: "C++"
#                                                  }))

#     with pytest.raises(openai.BadRequestError):
#         _ = await client.completions.create(
#             model=MODEL_NAME,
#             prompt="Give an example string that fits this regex",
#             extra_body=dict(guided_regex=TEST_REGEX, guided_json=TEST_SCHEMA))


# @pytest.mark.parametrize("guided_decoding_backend",
#                          ["outlines", "lm-format-enforcer"])
# async def test_guided_choice_chat_logprobs(server, client: openai.AsyncOpenAI,
#                                            guided_decoding_backend: str):
#     messages = [{
#         "role": "system",
#         "content": "you are a helpful assistant"
#     }, {
#         "role":
#         "user",
#         "content":
#         "The best language for type-safe systems programming is "
#     }]
#     chat_completion = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         max_tokens=10,
#         logprobs=True,
#         top_logprobs=5,
#         extra_body=dict(guided_choice=TEST_CHOICE,
#                         guided_decoding_backend=guided_decoding_backend))
#     top_logprobs = chat_completion.choices[0].logprobs.top_logprobs

#     # -9999.0 is the minimum logprob returned by OpenAI
#     assert all(
#         isinstance(logprob, float) and logprob >= -9999.0
#         for token_dict in top_logprobs
#         for token, logprob in token_dict.items())


# async def test_response_format_json_object(server, client: openai.AsyncOpenAI):
#     for _ in range(2):
#         resp = await client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[{
#                 "role":
#                 "user",
#                 "content": ('what is 1+1? please respond with a JSON object, '
#                             'the format is {"result": 2}')
#             }],
#             response_format={"type": "json_object"})

#         content = resp.choices[0].message.content
#         loaded = json.loads(content)
#         assert loaded == {"result": 2}, loaded


# async def test_custom_role(server, client: openai.AsyncOpenAI):
#     # Not sure how the model handles custom roles so we just check that
#     # both string and complex message content are handled in the same way

#     resp1 = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[{
#             "role": "my-custom-role",
#             "content": "what is 1+1?",
#         }],  # type: ignore
#         temperature=0,
#         seed=0)

#     resp2 = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[{
#             "role": "my-custom-role",
#             "content": [{
#                 "type": "text",
#                 "text": "what is 1+1?"
#             }]
#         }],  # type: ignore
#         temperature=0,
#         seed=0)

#     content1 = resp1.choices[0].message.content
#     content2 = resp2.choices[0].message.content
#     assert content1 == content2


# async def test_guided_grammar(server, client: openai.AsyncOpenAI):
#     simple_sql_grammar = """
# start: select_statement

# select_statement: "SELECT" column "from" table "where" condition

# column: "col_1" | "col_2"
# table: "table_1" | "table_2"
# condition: column "=" number

# number: "1" | "2"
# """

#     completion = await client.completions.create(
#         model=MODEL_NAME,
#         prompt=("Generate a sql state that select col_1 from "
#                 "table_1 where it is equals to 1"),
#         temperature=1.0,
#         max_tokens=500,
#         extra_body=dict(guided_grammar=simple_sql_grammar))

#     content = completion.choices[0].text

#     # use Lark to parse the output, and make sure it's a valid parse tree
#     from lark import Lark
#     parser = Lark(simple_sql_grammar)
#     parser.parse(content)

#     # remove spaces for comparison b/c we removed them in the grammar
#     ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(" ", "")

#     assert content.strip() == ground_truth


# @pytest.mark.parametrize(
#     # first test base model, then test loras
#     "model_name",
#     [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
# )
# async def test_echo_logprob_completion(server, client: openai.AsyncOpenAI,
#                                        model_name: str):
#     tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
#     # test using text and token IDs
#     for prompt in ("Hello, my name is", [0, 0, 0, 0, 0]):
#         completion = await client.completions.create(model=model_name,
#                                                      prompt=prompt,
#                                                      max_tokens=5,
#                                                      temperature=0.0,
#                                                      echo=True,
#                                                      logprobs=1)

#         prompt_text = tokenizer.decode(prompt) if isinstance(prompt,
#                                                              list) else prompt
#         assert (completion.choices[0].text is not None
#                 and re.search(r"^" + prompt_text, completion.choices[0].text))
#         logprobs = completion.choices[0].logprobs
#         assert logprobs is not None
#         assert len(logprobs.text_offset) > 5
#         assert (len(logprobs.token_logprobs) > 5
#                 and logprobs.token_logprobs[0] is None)
#         assert (len(logprobs.top_logprobs) > 5
#                 and logprobs.top_logprobs[0] is None)
#         assert len(logprobs.tokens) > 5


# async def test_long_seed(server, client: openai.AsyncOpenAI):
#     for seed in [
#             torch.iinfo(torch.long).min - 1,
#             torch.iinfo(torch.long).max + 1
#     ]:
#         with pytest.raises(BadRequestError) as exc_info:
#             await client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[{
#                     "role": "system",
#                     "content": "You are a helpful assistant.",
#                 }],
#                 temperature=0,
#                 seed=seed)

#         assert ("greater_than_equal" in exc_info.value.message
#                 or "less_than_equal" in exc_info.value.message)


# async def test_extra_fields(self):
# 	try:
# 		await self.client.chat.completions.create(
# 			model=self.model_name,
# 			messages=[{
# 				"role": "system",
# 				"content": "You are a helpful assistant.",
# 				"extra_field": "0",
# 			}],  # type: ignore
# 			temperature=0,
# 			seed=0)
# 	except BadRequestError as e:
# 		assert "extra_forbidden" in e.message
# 	else:
# 		pytest.fail("DID NOT RAISE <class 'openai.BadRequestError'>")


```

