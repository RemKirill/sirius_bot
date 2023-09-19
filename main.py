import math
from multiprocessing.dummy import Pool
from pathlib import Path
import psycopg2
from psycopg2 import Error
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import matplotlib

import telebot

from create_report import report

matplotlib.use('agg')

token = 'TOKEN'
bot = telebot.TeleBot(token)
url = 'https://api.telegram.org/bot'
CONTENT_TYPES = ["audio", "photo", "sticker", "video", "video_note", "voice", "location", "contact",
                 "new_chat_members", "left_chat_member", "new_chat_title", "new_chat_photo", "delete_chat_photo",
                 "group_chat_created", "supergroup_chat_created", "channel_chat_created", "migrate_to_chat_id",
                 "migrate_from_chat_id", 'animation']

id_course_having = [560, 584, 587, 613, 637, 638]

pool = Pool(20)

def executor(fu): #многопоточка для бота
    def run(*a, **kw):
        pool.apply_async(fu, a, kw)
    return run


def convertStr(s):
    try:
        ret = int(s)
    except ValueError:
        ret = float("nan")
    return ret


@bot.message_handler(commands=['start'])
@executor
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, напиши id курса, я пришлю тебе отчет.')


@bot.message_handler(commands=['help'])
@executor
def start_message(message):
    bot.send_message(message.chat.id, 'Я еще не написал help.')


@bot.message_handler(content_types=CONTENT_TYPES) #Если поступает не текст, то бот такое не обоработает
@executor
def cont_message(message):
    bot.send_message(message.chat.id, 'Прости, я не умею обрабатывать такой тип сообщений. '
                                      'Напиши id курса, я пришлю тебе отчет.')


@bot.message_handler(content_types=['text'])
@executor
def text_message(message):
    print(message.chat.id)
    print(message.text)
    s = convertStr(message.text)
    id_course_having = []
    try:
        connection = psycopg2.connect(user="postgres",
                                      password="password",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="courses_sirius")
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()

        cursor.execute(f"""SELECT id
                                           FROM course;""")

        id_course_having = cursor.fetchall()

        id_course_having = [int(tup[0]) for tup in id_course_having] #Проверяем какие курсы есть в БД

    except (Exception, Error) as error:
        print("Ошибка при работе с PostgreSQL", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
            #print("Соединение с PostgreSQL закрыто")

    if (len(id_course_having) == 0):
        bot.send_message(message.chat.id, 'Не удалось подключиться к базе данных.')
    else:
        if (math.isnan(s)): #Если текст не число
            bot.send_message(message.chat.id, 'Необходимо указать id курса для получения отчета.')
        elif (s in id_course_having): #Если такой id есть в БД
            bot.send_message(message.chat.id, 'Приступил к формированию отчета, подождите немного.')
            f = open(f'Course_{s}_report.html', 'w+')
            html_template = report(s)

            f.write(html_template)
            f.close()

            bot.send_document(message.chat.id, open(f'Course_{s}_report.html', 'r'))
            Path(f"Course_{s}_report.html").unlink()
        else: #Если такого id нет в БД
            bot.send_message(message.chat.id, 'Курса с таким id не найдено, попробуйте еще раз.')

'''@bot.message_handler(content_types=['document'])
@executor
def document_message(message):
    global data_module
    global data_element
    global id_course_having
    try:
        print(len(data_module))
        file_info = bot.get_file(message.document.file_id)
        doc_type = re.search(r'\.[^.]*$', file_info.file_path)
        if (doc_type.group(0) == '.csv'):
            downloaded_file = bot.download_file(file_info.file_path)
            s = str(downloaded_file, 'utf-8')
            data = StringIO(s)
            tmp = pd.read_csv(data)
            if (tmp.columns.tolist() == header_module):
                data_module = data_module.append(tmp)
                id_course_having = list(set(data_element['course_id']) & set(data_module['course_id']))
                bot.send_message(message.chat.id, 'Добавил данные о прогрессе по модулям курса.')
            elif (tmp.columns.tolist() == header_element):
                data_element = data_element.append(tmp)
                id_course_having = list(set(data_element['course_id']) & set(data_module['course_id']))
                bot.send_message(message.chat.id, 'Добавил данные о прогрессе по элементам курса.')
            else:
                bot.send_message(message.chat.id, 'Название столбцов не совпадает, проверьте корректность.')
        else:
            bot.send_message(message.chat.id, 'Файл должен быть в формате .csv.')
    except Exception:
        bot.send_message(message.chat.id, 'Что-то пошло не так, пришлите csv повторно.')
'''

bot.polling()