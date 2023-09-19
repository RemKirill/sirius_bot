import pandas as pd
import numpy as np
import statistics
import psycopg2
from pathlib import Path
from psycopg2 import Error
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import seaborn as sns
import matplotlib.pyplot as plt
import base64


def report(course_id):
    try:
        connection = psycopg2.connect(user="postgres",
                                      password="password",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="courses_sirius")
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()

        def sql_request(table_name, col):

            cursor.execute(f"""SELECT {','.join(col)}
                        FROM {table_name}
                        WHERE course_id = {course_id}::varchar;""")

            record = cursor.fetchall()

            return pd.DataFrame(np.array(record), columns=col)

        col_course_graph = ['from_module_id', 'to_module_id', 'type']

        course_graph = sql_request('course_graph', col_course_graph)
        copy_course_graph = course_graph.copy()

        col_course_module = ['id', 'course_id', 'percent_to_pass', 'is_advanced', 'progress_max',
                             'steps_max', 'tasks_max', 'type', 'level']

        course_module = sql_request('course_module', col_course_module)

        '''col_user_element_progress = ['id', 'user_id', 'course_id', 'course_module_id', 'course_element_type',
                                     'course_element_id', 'progress_current', 'is_achieved', 'tries_count',
                                     'module_progress_id', 'time_created', 'time_updated', 'time_closed',
                                     'time_achieved', 'time_started', 'achieve_reason']

        user_element_progress = sql_request('user_element_progress', col_user_element_progress)'''

        col_user_module_progress = ['id', 'user_id', 'course_id', 'course_module_id', 'progress_current',
                                    'progress_failed', 'steps_done', 'is_achieved', 'is_closed', 'course_progress_id',
                                    'time_created', 'time_updated', 'time_closed', 'time_achieved', 'time_unlocked',
                                    'tasks_done', 'time_done', 'is_done', 'achieve_reason']

        user_module_progress = sql_request('user_module_progress', col_user_module_progress)

        '''col_course_element = ['id', 'module_id', 'element_type', 'element_id', 'is_advanced', 'max_tries', 'score', 'position']

        cursor.execute(f"""SELECT {','.join(col_course_element)}
                                FROM course_element
                                WHERE module_id IN ({'::varchar,'.join(course_module['id'])}::varchar) and element_type = 'task';""")

        record = cursor.fetchall()

        course_element = pd.DataFrame(np.array(record), columns=col_course_element)'''

        module_level = []
        module_level_ordinary = []
        module_level_advanced = []

        module_level.append(  #Курсы этапа 0-- те, которые есть в столбце from, но нет в столбце to
            list(set(copy_course_graph['from_module_id']).difference(set(copy_course_graph['to_module_id']))))

        while len(copy_course_graph) != 0: #Формируем module_level-- list с этапами модулей
            t = []
            for i in module_level:
                for j in i:
                    tmp = copy_course_graph[copy_course_graph['from_module_id'] == j]['to_module_id']
                    copy_course_graph = copy_course_graph.drop(index=tmp.index)
                    for k in tmp.values:
                        if k not in copy_course_graph['to_module_id'].values:
                            t.append(k)
            module_level.append(t)

        for i in module_level: #Выбираем какие курсы ordinary, какие advanced
            t_ordinary = []
            t_advanced = []
            for j in i:
                if course_module[course_module['id'] == j]['type'].values == 'ordinary':
                    t_ordinary.append(j)
                elif course_module[course_module['id'] == j]['type'].values == 'advanced':
                    t_advanced.append(j)
            module_level_ordinary.append(t_ordinary)
            module_level_advanced.append(t_advanced)

        def progress_reached(module_level): #Первая метрика. На вход list с этапами модулей
            result_progress_reached = []

            t = 0

            for j in module_level[0]:
                user_pass_module = user_module_progress[(user_module_progress['course_module_id'] == str(j))
                                                        & (user_module_progress['is_achieved'] == 'True')]
                #Отбираем тех, у кого пройдены хотя бы модули 0 этапа
                max_for_module = float(course_module[course_module['id'] == j]['progress_max'].values)
                if (max_for_module != 0):
                    t = t + statistics.fmean(pd.to_numeric(user_pass_module['progress_current'])) / max_for_module

            result_progress_reached.append(t / len(module_level[0]))

            for i in module_level[1:]:
                t = 0
                for j in i:
                    user_pass_module = user_module_progress[user_module_progress['course_module_id'] == str(j)]
                    max_for_module = float(course_module[course_module['id'] == j]['progress_max'].values)
                    if (max_for_module != 0):
                        t = t + statistics.fmean(pd.to_numeric(user_pass_module['progress_current'])) / max_for_module
                        # Делим на число тех, кто дошел до этого модуля (внутри fmean)
                result_progress_reached.append(t / len(i))

            return result_progress_reached

        result_progress_reached = progress_reached(module_level)

        result_progress_reached_ordinary = progress_reached(module_level_ordinary)

        def progress_all(module_level): #Вторая метрика. На вход list с этапами модулей
            result_progress_all = []

            t = 0

            for j in module_level[0]:
                user_pass_module = user_module_progress[(user_module_progress['course_module_id'] == str(j))
                                                        & (user_module_progress['is_achieved'] == 'True')]
                # Отбираем тех, у кого пройдены хотя бы модули 0 этапа
                max_for_module = float(course_module[course_module['id'] == j]['progress_max'].values)
                if (max_for_module != 0):
                    t = t + statistics.fmean(pd.to_numeric(user_pass_module['progress_current'])) / max_for_module

            user_take_0_level = user_pass_module.loc[user_pass_module['course_module_id'].isin(module_level[0])]

            count_take_0_level = len(user_take_0_level[user_take_0_level['is_achieved'] == 'True'])

            result_progress_all.append(t / len(module_level[0]))

            for i in module_level[1:]:
                t = 0
                for j in i:
                    user_pass_module = user_module_progress[user_module_progress['course_module_id'] == str(j)]
                    max_for_module = float(course_module[course_module['id'] == j]['progress_max'].values)
                    if (max_for_module != 0):
                        tmp = sum(pd.to_numeric(user_pass_module['progress_current'])) / count_take_0_level
                        # Делим на число тех, кто прошел модули этапа 0
                        t = t + tmp / max_for_module
                result_progress_all.append(t / len(i))

            return result_progress_all

        result_progress_all = progress_all(module_level)

        result_progress_all_ordinary = progress_all(module_level_ordinary)

        def count_all(module_level): #вспомогательная метрика. На вход list с этапами модулей

            result_count_all = []

            for i in module_level:
                t = []
                for j in i:
                    user_pass_module = user_module_progress[(user_module_progress['course_module_id'] == str(j))
                                                            & (user_module_progress['is_achieved'] == 'True')]
                    # Отбираем тех, у кого пройдены хотя бы модули 0 этапа
                    t.append(len(user_pass_module))
                result_count_all.append(statistics.fmean(t))

            return [i/max(result_count_all) for i in result_count_all]

        result_count_all = count_all(module_level)

        result_count_all_ordinary = count_all(module_level_ordinary)

        def progress_tasks(module_level): #Четвертая метрика. На вход list с этапами модулей

            result_progress_tasks = []

            t = 0

            for j in module_level[0]:
                user_pass_module = user_module_progress[(user_module_progress['course_module_id'] == str(j))
                                                        & (user_module_progress['is_achieved'] == 'True')]
                # Отбираем тех, у кого пройдены хотя бы модули 0 этапа
                max_for_module = float(course_module[course_module['id'] == j]['progress_max'].values)
                if (max_for_module != 0):
                    t = t + statistics.fmean(pd.to_numeric(user_pass_module['tasks_done'])) / max_for_module

            result_progress_tasks.append(t / len(module_level[0]))

            for i in module_level[1:]:
                t = 0
                for j in i:
                    user_pass_module = user_module_progress[user_module_progress['course_module_id'] == str(j)]
                    max_for_module = float(course_module[course_module['id'] == j]['progress_max'].values)
                    if (max_for_module != 0):
                        t = t + statistics.fmean(pd.to_numeric(user_pass_module['tasks_done'])) / max_for_module
                result_progress_tasks.append(t / len(i))

            return result_progress_tasks

        result_progress_tasks = progress_tasks(module_level)

        result_progress_tasks_ordinary = progress_tasks(module_level_ordinary)

        '''result_tries_count = []

        for i in module_level:
            t = 0
            for j in i:
                #print(j)
                user_pass_element = user_element_progress[(user_element_progress['course_module_id'] == str(j))
                                                          & (user_element_progress['is_achieved'] == 'True')
                                                          & (user_element_progress['course_element_type'] == 'task')]
                c_tasks = []
                for k in range(len(user_pass_element['course_element_id'])):
                    #
                    #if (j == '5022'):
                        #print(k)
                        #print(course_element[course_element['element_id']
                        #           == user_pass_element['course_element_id'].iloc[k]]['score'][0])
                    #print(float(course_element[course_element['element_id']
                     #              == user_pass_element['course_element_id'].iloc[k]]['score']))
                    tmp = float(course_element[course_element['element_id']
                                   == user_pass_element['course_element_id'].iloc[k]]['score'].iloc[0])
                    if tmp != 0:
                        #print(k)
                        #print(type(user_pass_element['progress_current'].iloc[k]))
                        #print(type(float(user_pass_element['progress_current'].iloc[k])))
                        c_tasks.append(float(user_pass_element['progress_current'].iloc[k])/tmp)
                if (len(c_tasks) != 0):    #если модуль чисто из видео или текста
                    t = t + statistics.fmean(c_tasks)
            result_tries_count.append(t / len(i))

        print(result_tries_count)'''

        eps = 0.05

        def mini_report_metrics_1_3(result): #Формирование мини отчета для метрик 1 и 3. На вход list с результатом
            m = min(result)
            res = []
            res_index = []
            for i in range(len(result)):
                if (result[i] - m) <= eps:
                    res_index.append(i)
                    res.append(result[i])
            if (res == result):
                s = f'''С точки зрения этой метрики поведение пользователей стабильно для этого курса, 
                так как все значения метрики не отличаются от минимального более чем на {str(eps)}'''
            else:
                if (len(res_index) == 1):
                    s = f'''Активность пользователей сильнее всего проседает на {str(res_index[0])} этапe. 
                    При этом максимум наблюдается на этапе {str(result.index(max(result)))}.'''
                else:
                    s = f'''Активность пользователей сильнее всего проседает на 
                    {', '.join(str(x) for x in res_index)} этапах. 
                    При этом максимум наблюдается на этапе {str(result.index(max(result)))}.'''
                if (result.index(max(result)) == 0):
                    s = s + ' Что может говорить о наибольшей заинтересованости слушателей в самом начале курса.'
            return s

        rep_progress_reached = mini_report_metrics_1_3(result_progress_reached)

        rep_progress_reached_ordinary = mini_report_metrics_1_3(result_progress_reached_ordinary)

        rep_progress_tasks = mini_report_metrics_1_3(result_progress_tasks)

        rep_progress_tasks_ordinary = mini_report_metrics_1_3(result_progress_tasks_ordinary)

        def non_increasing(L):
            return all(x >= y for x, y in zip(L, L[1:]))

        def mini_report_metrics_2(result): #Формирование мини отчета для метрики 2. На вход list с результатом
            if (non_increasing(result)):
                level_number = [j for i, j in zip(result, result[1:]) if (i - j) <= eps]
                if (len(level_number) == 0):
                    s = f'''Метрика является не возрастающей для данного курса.
                    При этом падение метрики значимо на каждом следуюшем этапе'''
                else:
                    s = f'''Метрика является не возрастающей для данного курса.
                        При этом начиная с этапа {str(result.index(level_number[0]))} 
                        различия метрики становятся незначительны. Метрика становится стабильной.'''
            else:
                jump = [j for i, j in zip(result, result[1:]) if j > i]
                if (len([j for i, j in zip(result, result[1:]) if j > i]) == 1):
                    s = f'''Метрика не является монотонной, 
                        наблюдается скачок активности на {str(result.index(jump[0]))} этапе.'''
                else:
                    s = f'''Метрика не является монотонной, 
                        наблюдается скачок активности на {', '.join(str(result.index(x)) for x in jump)} этапах.'''
            return s

        rep_progress_all = mini_report_metrics_2(result_progress_all)

        rep_progress_all_ordinary = mini_report_metrics_2(result_progress_all_ordinary)

        module_level_string = [',\n'.join(lst) for lst in module_level]

        def png_create(res, res_name, label):
            #на вход list с результатом, str с названием будущего файла, str с названием метрики
            dframe = pd.DataFrame(list(zip(module_level_string, res)),  # Переписывать
                                  columns=['module_level', f'''{res_name}'''])
            fig, axes = plt.subplots(figsize=(15, 10))
            for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                label.set_fontsize(18)
            g = sns.lineplot(data=dframe, x="module_level", y=f'''{res_name}''')
            g.axes.set_ylim(0, 1)
            plt.tight_layout()
            plt.title('', fontsize=21)
            plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
            plt.ylabel(f'''{label}''', fontsize=21)
            plt.savefig(f'''{res_name}.png''')

        png_create(result_progress_reached, 'result_progress_reached',
                   'Текущий прогресс дошедших')
        png_create(result_progress_all, 'result_progress_all',
                   'Текущий прогресс всех')
        png_create(result_count_all, 'result_count_all',
                   'Доля задач, решенных на max баллов')
        png_create(result_progress_tasks, 'result_progress_tasks',
                   'Доля задач, решенных на max баллов')
        png_create(result_progress_reached_ordinary, 'result_progress_reached_ordinary',
                   'Текущий прогресс всех для ordinary модулей')
        png_create(result_progress_all_ordinary, 'result_progress_all_ordinary',
                   'Текущий прогресс всех для ordinary модулей')
        png_create(result_count_all_ordinary, 'result_count_all_ordinary',
                   'Доля задач, решенных на max баллов для ordinary модулей')
        png_create(result_progress_tasks_ordinary, 'result_progress_tasks_ordinary',
                   'Доля задач, решенных на max баллов для ordinary модулей')

        '''dframe = pd.DataFrame(list(zip(module_level_string, result_progress_reached)), # Переписывать
                              columns=['module_level', 'result_progress_reached'])
        fig, axes = plt.subplots(figsize=(15, 10))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(18)
        g = sns.lineplot(data=dframe, x="module_level", y="result_progress_reached")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('', fontsize=21)
        plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
        plt.ylabel('Текущий прогресс дошедших', fontsize=21)
        plt.savefig('result_progress_reached.png')

        dframe = pd.DataFrame(list(zip(module_level_string, result_progress_all)),
                              columns=['module_level', 'result_progress_all'])
        fig, axes = plt.subplots(figsize=(15, 10))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(18)
        g = sns.lineplot(data=dframe, x="module_level", y="result_progress_all")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('', fontsize=21)
        plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
        plt.ylabel('Текущий прогресс всех', fontsize=21)
        plt.savefig('result_progress_all.png')

        dframe = pd.DataFrame(list(zip(module_level_string, result_count_all)),
                              columns=['module_level', 'result_count_all'])
        fig, axes = plt.subplots(figsize=(15, 10))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(18)
        g = sns.lineplot(data=dframe, x="module_level", y="result_count_all")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('', fontsize=21)
        plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
        plt.ylabel('Доля задач, решенных на max баллов', fontsize=21)
        plt.savefig('result_count_all.png')

        dframe = pd.DataFrame(list(zip(module_level_string, result_progress_tasks)),
                              columns=['module_level', 'result_progress_tasks'])
        fig, axes = plt.subplots(figsize=(15, 10))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(18)
        g = sns.lineplot(data=dframe, x="module_level", y="result_progress_tasks")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('', fontsize=21)
        plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
        plt.ylabel('Доля задач, решенных на max баллов', fontsize=21)
        plt.savefig('result_progress_tasks.png')

        dframe = pd.DataFrame(list(zip(module_level_string, result_progress_reached_ordinary)),
                              columns=['module_level', 'result_progress_reached_ordinary'])
        fig, axes = plt.subplots(figsize=(15, 10))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(18)
        g = sns.lineplot(data=dframe, x="module_level", y="result_progress_reached_ordinary")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('', fontsize=21)
        plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
        plt.ylabel('Текущий прогресс дошедших для ordinary модулей', fontsize=21)
        plt.savefig('result_progress_reached_ordinary.png')

        dframe = pd.DataFrame(list(zip(module_level_string, result_progress_all_ordinary)),
                              columns=['module_level', 'result_progress_all_ordinary'])
        fig, axes = plt.subplots(figsize=(15, 10))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(18)
        g = sns.lineplot(data=dframe, x="module_level", y="result_progress_all_ordinary")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('', fontsize=21)
        plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
        plt.ylabel('Текущий прогресс всех для ordinary модулей', fontsize=21)
        plt.savefig('result_progress_all_ordinary.png')

        dframe = pd.DataFrame(list(zip(module_level_string, result_count_all_ordinary)),
                              columns=['module_level', 'result_count_all_ordinary'])
        fig, axes = plt.subplots(figsize=(15, 10))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(18)
        g = sns.lineplot(data=dframe, x="module_level", y="result_count_all_ordinary")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('', fontsize=21)
        plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
        plt.ylabel('Доля задач, решенных на max баллов', fontsize=21)
        plt.savefig('result_count_all_ordinary.png')

        dframe = pd.DataFrame(list(zip(module_level_string, result_progress_tasks_ordinary)),
                              columns=['module_level', 'result_progress_tasks_ordinary'])
        fig, axes = plt.subplots(figsize=(15, 10))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(18)
        g = sns.lineplot(data=dframe, x="module_level", y="result_progress_tasks_ordinary")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('', fontsize=21)
        plt.xlabel('id модулей, сгруппированные по этапам', fontsize=21)
        plt.ylabel('Доля задач, решенных на max баллов для ordinary модулей', fontsize=21)
        plt.savefig('result_progress_tasks_ordinary.png')'''

        '''dframe = pd.DataFrame(list(zip([',\n'.join(lst) for lst in module_level], result_tries_count)),
                              columns=['module_level', 'result_tries_count'])
        fig, axes = plt.subplots(figsize=(3, 2))
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(6)
        g = sns.lineplot(data=dframe, x="module_level", y="result_tries_count")
        g.axes.set_ylim(0, 1)
        plt.tight_layout()
        plt.title('Example Plot', fontsize=7)
        plt.xlabel('Sepal Length', fontsize=7)
        plt.ylabel('Sepal Width', fontsize=7)
        plt.savefig('result_tries_count.svg')'''

        def img_to_base64(s): #При пересылке в tg картинки приходится кодировать в base64
            image = open(f'{s}.png', 'rb')
            image_read = image.read()
            return str(base64.b64encode(image_read))[2:(-1)]

        a = f'''
        <html>
        <head>
        <meta charset=utf-8>
        </head>
        <body>
        <h2> Отчет по курсу {course_id} <h2>
        <details>
        <summary> Описание подхода</summary>
        <p> 
        Исключим из рассмотрения модули <em>autograde</em> и <em>section</em>.
        </p>
        <p>
        Введем понятие этап-- 0 этапом назовем все модули, которые доступны ученику при старте курса.
        </p>
        <p>
        1 этапом назовем все модули, которые открываются при прохождении модулей 0 этапа (не обязательно всех, 
        некотрого количества модулей).
        </p>
        <p>
        Рекурсивно определим n этап как все модули, которые открываются при прохождении модулей 0, 1,..., (n-1) этапов.
        </p>
        <br>
        <p>
        Рассмотрим всех пользователей, которые завершили хотя бы один из модулей, 
        иначе считаем их случайными слушателями и не учитываем их.
        </p>
        <br>
        <p>
        Так как не все ученики проходят advanced модули. То будем представлять два набора графиков: 
        с информацией по всем модулям и отдельно только по ordinary модулям.
        </p>
        </details>
        <hr>
        <p>
        Рассмотрим первую метрику активности-- текущий прогресс пользователей, на i этапе курса, среди тех, 
        у кого данный модуль открыт.
        </p>
        <br>
        <p>
        Tекущим прогрессом для модуля назовем среднее по всем пользователям, 
        у которых данный модуль открыт, (<em>progress_current/progress_max</em> для модуля).
        </p>
        <p>
        Текущим прогрессом для i го этапа назовем среднее арифмитеческое текущих прогрессов всех модулей, 
        входящих в этот этап.
        </p>
        <br>
        <p>
        Данная метрика позволяет отследить результаты активности самых лояльных слушателей.
        </p>
        <br>
        <p>
        Далее представлены график и таблица для нашей метрики по всем этапам курса.
        </p>
        <br>
        <img src="data:image/png;base64,{img_to_base64('result_progress_reached')}" width="700", height = 500>
        <br>
        <br>
        {pd.DataFrame({'module_id':[', '.join(lst) for lst in module_level], 
                       'metric': [round(x, 3) for x in result_progress_reached]}).transpose().to_html()}
        <br>
        <p>{rep_progress_reached}</p>
        <details>
        <summary> Метрика для ordinary модулей</summary>
        <br>
        <img src="data:image/png;base64,{img_to_base64('result_progress_reached_ordinary')}" width="700", height = 500>
        <br>
        <br>
        {pd.DataFrame({'module_id':[', '.join(lst) for lst in module_level], 
                       'metric': [round(x, 3) for x in result_progress_reached_ordinary]}).transpose().to_html()}
        <p>{rep_progress_reached_ordinary}</p>
        <br>
        </details>
        <hr>
        <p>
        Рассмотрим вторую метрику активности-- текущий прогресс пользователей, на i этапе курса, 
        среди слушателей начавших курс.
        </p>
        <br>
        <p>
        Текущим прогрессом для модуля назовем среднее по всем пользователям, 
        прошедших хотя бы нулевой этап, (<em>progress_current/progress_max</em> для модуля).
        </p>
        <p>
        Текущим прогрессом для i го этапа назовем среднее арифмитеческое текущих прогрессов всех модулей, 
        входящих в этот этап.
        </p>
        <br>
        <p>
        Данная метрика позволяет посмотреть на динамику по всем слушателям, в том числе и тем, 
        кто приостановил обучение на курсе.
        </p>
        <br>
        <p>
        Далее представлены график и таблица для нашей второй метрики по всем этапам курса.
        </p>
        <br>
        <img src="data:image/png;base64,{img_to_base64('result_progress_all')}" width="700", height = 500>
        <br>
        <br>
        {pd.DataFrame({'module_id':[', '.join(lst) for lst in module_level], 
                       'metric': [round(x, 3) for x in result_progress_all]}).transpose().to_html()}
        <br>
        <p>{rep_progress_all}</p>
        <details>
        <summary> Метрика для ordinary модулей</summary>
        <img src="data:image/png;base64,{img_to_base64('result_progress_all_ordinary')}" width="700", height = 500>
        <br>
        <br>
        {pd.DataFrame({'module_id':[', '.join(lst) for lst in module_level], 
                       'metric': [round(x, 3) for x in result_progress_all_ordinary]}).transpose().to_html()}
        <br>
        <p>{rep_progress_all_ordinary}</p>
        </details>
        <hr>
        <p>
        Рассмотрим третью метрику активности-- доля задач, решенных в данном модуле на максимальное количетсво баллов
         для дошедших до данного модуля.
        </p>
        <br>
        <p>
        Долей решенных на максимум задач для модуля назовем среднее по всем пользователям, 
        у которых данный модуль открыт, (<em>tasks_done/progress_max</em> для модуля).
        </p>
        <p>
        Долей решенных на максимум задач для i го этапа назовем среднее арифмитеческое текущих прогрессов всех модулей, 
        входящих в этот этап.
        </p>
        <br>
        <p>
        Данная метрика показывает насколько часто слушатели стремятся закрыть задачи на максимальный балл.
        </p>
        <br>
        <p>
        Далее представлены график и таблица для нашей метрики по всем этапам курса.
        </p>
        <br>
        <img src="data:image/png;base64,{img_to_base64('result_progress_tasks')}" width="700", height = 500>
        <br>
        <br>
        {pd.DataFrame({'module_id':[', '.join(lst) for lst in module_level], 
                       'metric': [round(x, 3) for x in result_progress_tasks]}).transpose().to_html()}
        <br>
        <p>{rep_progress_tasks}</p>
        <details>
        <summary> Метрика для ordinary модулей</summary>
        <img src="data:image/png;base64,{img_to_base64('result_progress_tasks_ordinary')}" width="700", height = 500>
        <br>
        <br>
        {pd.DataFrame({'module_id':[', '.join(lst) for lst in module_level], 
                       'metric': [round(x, 3) for x in result_progress_tasks_ordinary]}).transpose().to_html()}
        <br>
        <p>{rep_progress_tasks_ordinary}</p>
        </details>
                <hr>
        <p>
        Рассмотрим вспомогательную метрику активности-- количество пользователей, прошедших i этап курса, 
        среди слушателей начавших курс.
        </p>
        <br>
        <p>
        Количество пользователей для модуля-- количество тех, кто прошел данный модуль.
        </p>
        <p>
        Количество пользователей для i го этапа назовем среднее арифмитеческое Количество пользователей всех модулей, 
        входящих в этот этап.
        </p>
        <br>
        <p>
        Данная метрика является вспомогательной при анализае предыдуших.
        </p>
        <br>
        <p>
        Далее представлены график и таблица для вспомогательной метрики по всем этапам курса.
        </p>
        <br>
        <img src="data:image/png;base64,{img_to_base64('result_count_all')}" width="700", height = 500>
        <br>
        <br>
        {pd.DataFrame({'module_id':[', '.join(lst) for lst in module_level], 
                       'metric': [round(x, 3) for x in result_count_all]}).transpose().to_html()}
        <br>
        <details>
        <summary> Метрика для ordinary модулей</summary>
        <img src="data:image/png;base64,{img_to_base64('result_count_all_ordinary')}" width="700", height = 500>
        <br>
        <br>
        {pd.DataFrame({'module_id':[', '.join(lst) for lst in module_level], 
                       'metric': [round(x, 3) for x in result_count_all_ordinary]}).transpose().to_html()}
        <br>
        </details>
        <p>Итоговые выводы должны быть сделаны человеком.</p>
        </body>
        </html>
        '''

        plt.close('all')
        Path("result_progress_reached.png").unlink()
        Path("result_progress_all.png").unlink()
        Path("result_count_all.png").unlink()
        Path("result_progress_tasks.png").unlink()
        Path("result_progress_reached_ordinary.png").unlink()
        Path("result_progress_all_ordinary.png").unlink()
        Path("result_count_all_ordinary.png").unlink()
        Path("result_progress_tasks_ordinary.png").unlink()

        return a

    except (Exception, Error) as error:
        print("Ошибка при работе с PostgreSQL", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
            # print("Соединение с PostgreSQL закрыто")
