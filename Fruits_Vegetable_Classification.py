import streamlit as st
from PIL import Image
#from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

model = load_model('FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

fruits_chinese = {
    'Apple': '苹果',
    'Banana': '香蕉',
    'Bell Pepper': '甜椒',
    'Chilli Pepper': '辣椒',
    'Grapes': '葡萄',
    'Jalapeño': '哈拉佩诺椒',
    'Kiwi': '奇异果',
    'Lemon': '柠檬',
    'Mango': '芒果',
    'Orange': '橙子',
    'Paprika': '灯笼椒',
    'Pear': '梨',
    'Pineapple': '菠萝',
    'Pomegranate': '石榴',
    'Watermelon': '西瓜'
}

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

chinese_vegetables = ['甜菜根', '包心菜', '辣椒', '胡萝卜', '花椰菜', '玉米', '黄瓜', '茄子', '生姜',
                      '生菜', '洋葱', '豌豆', '土豆', '小红萝卜', '大豆', '菠菜', '甜玉米', '甘薯',
                      '番茄', '芜菁']

# 使用zip()函数将两个列表组合起来并转换为字典
vegetable_dict = dict(zip(vegetables, chinese_vegetables))



def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    st.title("蔬菜水果分类检测")
    img_file = st.file_uploader("上传一张图片", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result = processed_img(save_image_path)
            print(result)
            if result in vegetables:
                st.info('**类目 : 蔬菜**')
                try:
                    veg_result = vegetable_dict[result]
                    st.success(f"**识别 : {veg_result}**")
                except Exception as e:
                    st.error(f"识别出错: {e}")
            else:
                st.info('**类目 : 水果**')
                try:
                    fruits_chinese_result = fruits_chinese[result]
                    st.success(f"**识别 : {fruits_chinese_result}**")
                except Exception as e:
                    st.error(f"识别出错: {e}")


            #st.success("**识别 : " + vegetable_dict(result) + '**')
            #cal = fetch_calories(result)
            #if cal:
            #    st.warning('**' + cal + '(100 grams)**')


run()
