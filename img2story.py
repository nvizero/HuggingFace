



from transformers import pipeline
from langchain import PromptTemplate , LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os


def img2text(url):
  image_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
  text = image_to_text(url)[0]["generated_text"]
  print(text)
  return text

def generate_story(scenario):
  template = """
  你有一個窮光頭,請你以這句話來說一個中文的故事,以悲傷做結尾 
  CONTEXT: {scenario}
  STORY:
  """
  prompt = PromptTemplate(template=template, input_variables=["scenario"])

  # Create an instance of ChatOpenAI and use it for LLMChain
  openai_instance = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1, openai_api_key='openai  的API KEY 這個要付費使用.')
  # openai  的API KEY 這個要付費使用.
  # 請至https://platform.openai.com/overview 
  story_llm = LLMChain(llm=openai_instance, prompt=prompt, verbose=True)
  story = story_llm.predict(scenario=scenario)
  print(story)
  return story

 


 
scenario = img2text("aa.png")
story = generate_story(scenario)

 




