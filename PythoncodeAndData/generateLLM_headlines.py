import openai
import time

openai.api_key = "<API KEY>"

def generate_headlines(batch_size=50, total=2000, output_file="sesotho_llm_headlinesCGPT3.5.txt"):
    generated = 0
    with open(output_file, "w", encoding="utf-8") as f:
        while generated < total:
            try:
                prompt = (
                    f"Generate {batch_size} unique Sesotho news headlines. "
                    "Each headline must be realistic, machine-generated, and wristten in the Sesotho language. "
                    "Output them one per line. Do not number them or include anything else."
                )

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that writes realistic Sesotho news headlines."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9,
                    max_tokens=1000,
                )

                headlines = response['choices'][0]['message']['content'].splitlines()
                headlines = [hl.strip() for hl in headlines if hl.strip() and not hl.strip().isdigit()]

                for hl in headlines:
                    f.write(f"{hl}\n1\n")
                    generated += 1
                    if generated >= total:
                        break

                print(f"Generated {generated}/{total} headlines...")

            except Exception as e:
                print(f"Error occurred: {e}")

    print("Done!")

generate_headlines()