
# vLLM Test Worker Deployment

This repository contains the `vLLM Worker` deployment along with a comprehensive suite of tests to ensure the proper functionality of the deployment. The following tests are included in this suite:

## Tests Overview

1. **test_check_models**: Validates the availability and correctness of models.
2. **test_single_completion**: Tests a single completion request.
3. **test_single_chat_session**: Verifies a single chat session.
4. **test_completion_streaming**: Ensures streaming functionality for completion.
5. **test_chat_streaming**: Ensures streaming functionality for chat.
6. **test_complex_message_content**: Tests handling of complex message content.
7. **test_extra_fields**: Checks the handling of extra fields in requests.
8. **test_guided_choice_completion**: Validates guided choice completion functionality.
9. **test_batch_completions**: Verifies batch completion requests.
10. **test_guided_decoding_type_error**: Checks for type errors in guided decoding.
11. **test_custom_role**: Validates custom role functionality.
12. **test_zero_logprobs**: Tests zero log probabilities functionality.

## Running the Tests

To run the tests, follow the steps below:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/vllm-worker-deployment.git
    cd vllm-worker-deployment
    ```

2. **Install the Dependencies**:
    Ensure you have Python and pip installed. Then, install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**:
    Configure the necessary environment variables for your deployment, create .env file and add:
    ```sh
    RUNPOD_API_KEY="<YOUR_API_KEY>"
    ```

4. **Run the Test Suite**:
    Use the following command to run all tests:
    ```sh
    python run.py
    ```

