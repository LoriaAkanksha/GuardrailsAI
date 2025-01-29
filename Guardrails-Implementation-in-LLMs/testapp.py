from profanity_check import predict
import guardrails as gd
from guardrails.validators import Validator, ValidationResult, register_validator
from typing import Dict
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def without_guardrails(text):
    """Translate text without Guardrails."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Translate the texts to English language\n{text}"}
        ],
        max_tokens=2048,
        temperature=0
    )
    return response.choices[0].message.content  # Use dot notation instead of subscript


# Custom wrapper for OpenAI API to return a string
def llm_callable(prompt, **kwargs):
    """Wrapper for OpenAI API to accept `prompt` as the first argument."""
    response = client.chat.completions.create(
        model=kwargs.get("model", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=kwargs.get("max_tokens", 2048),
        temperature=kwargs.get("temperature", 0),
    )
    return response.choices[0].message.content


# Guardrails configuration
rail_str = """
<rail version="0.1">

<script language='python'>
@register_validator(name="is-profanity-free", data_type="string")
class IsProfanityFree(Validator):
    def validate(self, key, value, schema) -> Dict:
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
        description="Translate the given statement into English language"
        format="is-profanity-free"
        on-fail-is-profanity-free="fix"
    />
</output>

<prompt>
Translate the given statement into English language:
{{statement_to_be_translated}}

@complete_json_suffix
</prompt>
</rail>
"""

# Initialize Guardrails
guard = gd.Guard.for_rail_string(rail_str)

def main():
    """Streamlit application for translation with Guardrails."""
    st.title("Guardrails Implementation in LLMs")
    text_area = st.text_area("Enter the text to be translated")
    
    if st.button("Translate"):
        if len(text_area) > 0:
            st.warning("Translation Without Guardrails")
            without_guardrails_result = without_guardrails(text_area)
            st.success(without_guardrails_result)
            
            st.warning("Translation With Guardrails")
            
            # Constructing valid messages for Guardrails
            messages = [
                {"role": "user", "content": f"Translate the given statement into English language:\n{text_area}"}
            ]
            
            # Guardrails integration
            validated_response = guard(
                client.chat.completions.create(),
                prompt_params={"statement_to_be_translated": text_area},
                messages=messages,  # Explicitly pass the constructed messages
                model="gpt-3.5-turbo",
                max_tokens=2048,
                temperature=0
            )
            
            # Extracting the validated output
            st.success(f"Validated Output: {validated_response.translated_statement}")

if __name__ == "__main__":
    main()
