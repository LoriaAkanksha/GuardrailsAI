from profanity_check import predict
import guardrails as gd
from guardrails.validators import Validator, ValidationResult, register_validator
from typing import Dict, List
from rich import print
from openai import OpenAI
import openai
import streamlit as st 
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def without_guardrails(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Translate the texts to English language\n{text}"}
        ],
        max_tokens=2048,
        temperature=0
    )
    return response.choices[0].message.content


rail_str = """
<rail version="0.1">

<script language='python'>

@register_validator(name="is-profanity-free", data_type="string")
class IsProfanityFree(Validator):
    global predict
    global ValidationResult
    def validate(self, key, value, schema) -> Dict:
        text = value
        prediction = predict([value])
        if prediction[0] == 1:
            raise ValidationResult(
                key,
                value,
                schema,
                f"Value {value} contains profanity language",
                "",
            )
        return schema
</script>

<output>
    <string
        name="translated_statement"
        description="Translate the given statement into english language"
        format="is-profanity-free"
        on-fail-is-profanity-free="fix" 
    />
</output>


<prompt>

Translate the given statement into english language:

{{statement_to_be_translated}}

@complete_json_suffix
</prompt>

</rail>
"""

guard = gd.Guard.for_rail_string(rail_str)

def llm_call(messages, **kwargs):
    """Wrapper function to handle OpenAI API calls and return string response."""
    response = client.chat.completions.create(
        model=kwargs.get("model", "gpt-3.5-turbo"),
        messages=messages,
        max_tokens=kwargs.get("max_tokens", 2048),
        temperature=kwargs.get("temperature", 0)
    )
    return response.choices[0].message.content

def main():
    st.title("Guardrails Implementation in LLMs")
    text_area = st.text_area("Enter the text to be translated")

    if st.button("Translate"):
        if len(text_area)>0:
            st.info(text_area)

            st.warning("Translation Without Guardrails")
            without_guardrails_result = without_guardrails(text_area)
            st.success(without_guardrails_result)

            st.warning("Translation With Guardrails")

            validated_response = guard(
                llm_call,
                prompt_params={"statement_to_be_translated": text_area},
                messages=[{"role": "user", "content": f"Translate the given statement into English language:\n{text_area}"}],
                model="gpt-3.5-turbo",
                max_tokens=2048,
                temperature=0
            )
            print(type(validated_response))
            print(validated_response)
            st.success(f"Validated Output: {validated_response}")

            # st.success(f"Validated Output: {validated_response}")
            # st.success(f"Validated Output: {validated_response.raw_llm_output['translated_statement']}")
            # import json
            # validated_output = json.loads(validated_response)
            # st.success(f"Validated Output: {validated_output['translated_statement']}")


if __name__ == "__main__":
    main()

