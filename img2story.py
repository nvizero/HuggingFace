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

#從前有一只窮光頭，他是一隻名叫阿祥的卡通狗，經常帶著一張落寞的臉和一個大鼻子。阿祥在一個繁忙的城市生活著，但他總是望著高樓大廈和豪華汽車感到無奈。
#阿祥每天都在城市的垃圾堆中尋找剩食和棄物，用來填飽肚子。他連一張舒適柔軟的被子都沒有，只能躺在寒冷的水泥地上。儘管這樣的生活艱辛，阿祥總是堅強地面對。
#他希望有一天，能改變自己的命運，過上好日子。
#有一天，阿祥路過一家漂亮的咖啡館，透過窗戶，他看見了一架鋼琴。阿祥心中湧起了一絲渴望，他渴望能成為一位音樂家，彈奏美妙的樂曲。但窮光頭的他無力實現這個夢想。
#阿祥天天前往咖啡館靜靜地觀賞別人彈奏鋼琴。他喜歡閉上眼睛，聆聽著那些美妙的音符。即使身處貧困與絕望之中，音樂成了阿祥唯一的心靈寄託。
#某一天，阿祥在街上遇到了一位富有仁慈的男人，他傾聽了阿祥的心聲，並了解到他夢想彈奏鋼琴的心願。這位男人決心幫助阿祥實現夢想。他帶阿祥去了一間音樂學院，並安排了一位著名鋼琴教師教授阿祥彈奏。
#日復一日，阿祥全心投入學習琴藝，努力突破自己的極限。他的指尖跳舞著，帶著滄桑的眼神卻充滿了希望和熱情。
#然而，就在阿祥即將在音樂大賽中展示所學之時，一個悲劇發生了。他被一輛偷車賊趕上，搶走了他珍貴的鋼琴。阿祥的夢想瞬間破滅，他回到了原本貧困的生活中。
#阿祥彷彿身陷黑暗的深淵，他失去了自己的方向感。他黯然地逡巡在城市的街頭，看著其他人幸福快樂的生活。他明白自己注定只能是這個世界的一個窮光頭。
#然而，一天，當阿祥走在街上時，他突然聽到一段熟悉的旋律。他追著音樂的源頭，來到了一家鋼琴店。店主對他說，一位神秘的捐贈者贈送了一架全新的鋼琴，並指定要送給阿祥。
#阿祥感到猶如夢中，他坐在鋼琴前，彈奏著那些美妙的音符。他再度找回了追求音樂的熱情，並將這份溫暖和喜悅傳遞給了附近的居民。
#然而，阿祥的故事並非以快樂結尾。當他彈奏完最後一個音符時，他突然感到一陣暈眩，倒在了地上。醫生的診斷讓所有人傷心欲絕，阿祥患上了一個無法醫治的疾病。
#阿祥的生命逐漸走到了盡頭。在他離世之前，他再次彈奏了他最喜愛的樂曲，將所有的心情和夢想都融入其中。人們悲痛地目送著阿祥的離去，他們永遠不會忘記這只窮光頭，這位獻出了全部的靈魂和熱情的音樂家。
#阿祥的故事讓人們明白，有時即使我們擁有夢想，努力去實現它們，命運也可能不總是美好的。然而，生活中仍然有著美好的瞬間，即使在最黑暗的時刻也能照亮我們的心靈。
#阿祥的傳奇將永遠在人們心中流傳，成為勇敢追求夢想的象徵。



