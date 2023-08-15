from transformers import pipeline
from langchain import PromptTemplate , LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
"""

在colab要先裝以下套件

!pip install transformers
!pip install langchain

!pip install langchain==0.0.235 openai
!pip install matplotlib-venn
!apt-get -qq install -y libfluidsynth1
!pip install openai
"""
# yt 完整教學 https://www.youtube.com/watch?v=gkZvrPhzHRQ
def img2text(url):
  image_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
  text = image_to_text(url)[0]["generated_text"]
  print(text)
  return text

def generate_story(scenario):
  template = """
  一個窮光蛋,請你以這句話來說一個中文的故事,以悲傷做結尾 
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

# 以下是AI 產出的內容 
#> Finished chain.

#從來都沒有人記得這隻大鼻子的卡通狗的名字，他就像一個窮光蛋一般過著貧困的生活。他總是流連在大街小巷，找尋一些食物來度過他的每一天。然而，即使在最艱困的日子裡，這隻可憐的狗總是帶著微笑，
#透過他的大鼻子去尋找生活中的美好。
#
#有一天，卡通狗聽說城市劇院正在舉辦一場聖誕晚會，聚集了全城最有才華的明星和藝術家。卡通狗立刻心生一絲希望，他相信只要給予自己一個機會，他也能成為其中的一分子。
#
#他去尋找一個童星學校，希望能透過舞蹈或唱歌的演出來改變自己的命運。然而，不幸的是，他竟然聽到了一個殘酷的事實－他無法唱歌，也不擅長舞蹈。
#在這個時候，他感到自己的大鼻子變得沈重，他哀傷地離開了學校，對自己的命運感到絕望。
#回到寂寞的角落，卡通狗抱頭痛哭。他覺得自己再也無法改變這個貧困的命運，再也不可能有人注意到他。他感到孤獨而絕望。
#
#然而，正當所有的希望似乎都破滅之際，他注意到劇院投放了大量的廣告，尋找一隻外表奇特但具有真摯表演專注力的卡通狗。卡通狗的心再次燃起希望，他相信這次機會不能錯過。
#
#他決定前去試試，即使再大的鼻子也不再是他的阻礙。舞台燈光下，卡通狗發揮出他最真摯的演技，用他的大鼻子演繹著生命中的故事。觀眾無不被他的演出吸引住，留下令人驚嘆的掌聲。
#
#然而，就在他以為一切都會改變的時刻，一個突然的意外發生。卡通狗在舞台上跌倒，整個劇院變得寂靜無聲。大家都擔心他是否受傷，但更令人沮喪的是，他的大鼻子也因為摔倒而受到損傷。
#
#卡通狗在傷痛中站起來，在他失去了大鼻子的姿態下，他從劇院悲傷地離開。他知道他再也無法回到那個舞臺上，成為他夢寐以求的明星。這悲傷的結局，讓他深深地感受到了命運的殘酷。
#
#然而，傷痛並沒有讓卡通狗放棄，他決定繼續走下去，即使失去了大鼻子，也要找尋屬於自己的快樂。儘管他無法改變自己的命運，卻在失去中找到了生命中真正的價值，那就是堅強和勇氣。
#
#在每一片夜空下，這隻窮光蛋的狗總是默默地關注著每個過往者，儘管再貧困，再困難，卻依然保持著微笑，那是他用大鼻子找到的最珍貴的寶藏。



