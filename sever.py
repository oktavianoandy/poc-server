from flask import Flask, request, jsonify
from os.path import exists
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
import math
import csv
import json
import sqlite3
import socket
import cv2

RERATA = 0
BANYAK_BOX = 0
PERSENTASE_NILAI_NOL = 0
LABEL = ""
TOTAL_FRAME = 0
DURATION = 0

my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    data = {
        "username": "admin",
        "email": "admin@localhost",
    }

    return data


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    video_file = request.files['file']

    filename = secure_filename(video_file.filename)
    filePath = 'Uploads/' + filename

    video_file.save(filePath)

    print("\nReceived File name : " + filename)

    preprocessing(filePath)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%d/%m/%Y")

    data = {
        "rerata": str(RERATA),
        "banyak_box": str(BANYAK_BOX),
        "persentase": str(PERSENTASE_NILAI_NOL),
        "label": str(LABEL),
        "total_frame": str(TOTAL_FRAME),
        "duration": str(DURATION),
        "date_created": str(current_date),
        "time_created": str(current_time)
    }

    return data


def db_connection():
    conn = None
    try:
        conn = sqlite3.connect("Database/naive_bayes.sqlite")
    except sqlite3.error as e:
        print(e)
    return conn


def db_create_table():
    conn = db_connection()
    cursor = conn.cursor()

    sql_query = """ CREATE TABLE IF NOT EXISTS tbl_data_training (
        id integer PRIMARY KEY,
        nama_video text,
        rerata float,
        banyak_box float,
        persentase_nilai_nol float,
        label text
    )"""

    cursor.execute(sql_query)

    print("tbl_data_training table is created.")


def db_check_is_exist():
    file_exist = exists("Database/naive_bayes.sqlite")
    return file_exist


def db_import_data_training():
    conn = db_connection()
    cursor = conn.cursor()

    file = open('Dataset/data_training.csv')
    contents = csv.reader(file)

    sql_query = """INSERT INTO tbl_data_training 
        (nama_video, rerata, banyak_box, persentase_nilai_nol, label) 
        VALUES (?, ?, ?, ?, ?)
    """

    cursor.executemany(sql_query, contents)

    conn.commit()

    print('Success import data training from csv')


def db_get_all_data_training():
    conn = db_connection()
    cursor = conn.cursor()

    sql_query = "SELECT * FROM tbl_data_training"

    row = cursor.execute(sql_query).fetchall()

    for r in row:
        print(r)


def naive_bayes_process(rerata_testing, banyak_box_testing, persentase_testing):
    print('Menjalankan tahap klasifikasi naive bayes...')
    conn = db_connection()
    cursor = conn.cursor()

    # cari mean tiap atribut
    # cari mean atribut rerata
    mean_rerata_bagus = cari_mean_atribut('rerata', 'Bagus')
    mean_rerata_menuju_tidak_bagus = cari_mean_atribut('rerata', 'Tidak Bagus')

    # cari mean atribut banyak_box
    mean_banyak_box_bagus = cari_mean_atribut('banyak_box', 'Bagus')
    mean_banyak_box_tidak_bagus = cari_mean_atribut('banyak_box', 'Tidak Bagus')

    # cari mean atribut persentase_nilai_nol
    mean_persentase_nilai_nol_bagus = cari_mean_atribut('persentase_nilai_nol', 'Bagus')
    mean_persentase_nilai_nol_tidak_bagus = cari_mean_atribut('persentase_nilai_nol', 'Tidak Bagus')

    # cari standar deviasi tiap atribut
    # cari standar deviasi rerata
    std_rerata_bagus = cari_std_atribut('rerata', 'Bagus')
    std_rerata_tidak_bagus = cari_std_atribut('rerata', 'Tidak Bagus')

    # cari standar deviasi banyak_box
    std_banyak_box_bagus = cari_std_atribut('banyak_box', 'Bagus')
    std_banyak_box_tidak_bagus = cari_std_atribut('banyak_box', 'Tidak Bagus')

    # cari standar deviasi persentase_nilai_nol
    std_persentase_nilai_nol_bagus = cari_std_atribut('persentase_nilai_nol', 'Bagus')
    std_persentase_nilai_nol_tidak_bagus = cari_std_atribut('persentase_nilai_nol', 'Tidak Bagus')

    # cari nilai probabilitas tiap kelas
    p_kelas_bagus = cari_nilai_probabilitas_kelas('Bagus')
    p_kelas_tidak_bagus = cari_nilai_probabilitas_kelas('Tidak Bagus')

    # Menghitung nilai prediksi
    # prediksi rerata
    rerata_bagus = perhitungan_nilai_atribut_prediksi(std_rerata_bagus
                                                      , rerata_testing
                                                      , mean_rerata_bagus)

    rerata_tidak_bagus = perhitungan_nilai_atribut_prediksi(std_rerata_tidak_bagus
                                                            , rerata_testing
                                                            , mean_rerata_menuju_tidak_bagus)

    # prediksi banyak_box
    banyak_box_bagus = perhitungan_nilai_atribut_prediksi(std_banyak_box_bagus
                                                          , banyak_box_testing
                                                          , mean_banyak_box_bagus)

    banyak_box_tidak_bagus = perhitungan_nilai_atribut_prediksi(std_banyak_box_tidak_bagus
                                                                , banyak_box_testing
                                                                , mean_banyak_box_tidak_bagus)

    # prediksi persentase_nilai_nol
    persentase_nilai_nol_bagus = perhitungan_nilai_atribut_prediksi(std_persentase_nilai_nol_bagus
                                                                    , persentase_testing
                                                                    , mean_persentase_nilai_nol_bagus)

    persentase_nilai_nol_tidak_bagus = perhitungan_nilai_atribut_prediksi(std_persentase_nilai_nol_tidak_bagus
                                                                          , persentase_testing
                                                                          , mean_persentase_nilai_nol_tidak_bagus)

    # perhitungan akhir
    nilai_kelas_bagus = perhitungan_akhir(rerata_bagus
                                          , banyak_box_bagus
                                          , persentase_nilai_nol_bagus
                                          , p_kelas_bagus)

    nilai_kelas_tidak_bagus = perhitungan_akhir(rerata_tidak_bagus
                                                , banyak_box_tidak_bagus
                                                , persentase_nilai_nol_tidak_bagus
                                                , p_kelas_tidak_bagus)

    # perangkingan
    label = perankingan(nilai_kelas_bagus
                        , nilai_kelas_tidak_bagus)

    print('Semua proses berhasil telah dijalankan.')

    return label


def cari_mean_atribut(atribut, nilai_label):
    conn = db_connection()
    cursor = conn.cursor()

    sql_query = "SELECT avg(" + atribut + ") " \
                                          "FROM tbl_data_training " \
                                          "WHERE label = '" + nilai_label + "'"

    result = cursor.execute(sql_query).fetchone()[0]

    return result


def cari_std_atribut(atribut, nilai_label):
    conn = db_connection()
    cursor = conn.cursor()

    sql_query = "SELECT " + atribut + " FROM tbl_data_training WHERE label = '" + nilai_label + "'"

    row = cursor.execute(sql_query).fetchall()
    arr_std_rerata_bagus = []
    for r in row:
        arr_std_rerata_bagus.append(r[0])

    result = np.std(arr_std_rerata_bagus, ddof=1)

    return result


def cari_nilai_probabilitas_kelas(nilai_label):
    conn = db_connection()
    cursor = conn.cursor()

    sql_query = "SELECT count(label) FROM tbl_data_training"
    total_kelas = cursor.execute(sql_query).fetchone()[0]
    sql_query = "SELECT count(label) FROM tbl_data_training where label = '" + nilai_label + "'"
    jumlah_kelas_by_label = cursor.execute(sql_query).fetchone()[0]
    result = jumlah_kelas_by_label / total_kelas

    return result


def cari_nilai_probabilitas_atribut(nilai_label, atribut, nilai_atribut):
    # untuk mencari nilai probabilitas atribut non-numerik
    conn = db_connection()
    cursor = conn.cursor()

    sql_query = "SELECT count(label) FROM tbl_data_training WHERE " \
                "label = '" + nilai_label + "' AND " \
                + atribut + " = '" + nilai_atribut + "'"

    jumlah_atribut = cursor.execute(sql_query).fetchone()[0]

    sql_query = "SELECT count(label) FROM tbl_data_training where label = '" + nilai_label + "'"
    jumlah_kelas_by_label = cursor.execute(sql_query).fetchone()[0]

    result = jumlah_atribut / jumlah_kelas_by_label

    return result


def perhitungan_nilai_atribut_prediksi(std, nilai_atribut_testing, mean):
    result = 1 / np.sqrt(2 * math.pi * std) * np.exp(
        -((nilai_atribut_testing - mean) ** 2 / (2 * (std ** 2))))

    return result


def perhitungan_akhir(atr1, atr2, atr3, p_kelas):
    result = atr1 * atr2 * atr3 * p_kelas
    return result


def perankingan(nilai_kelas_bagus, nilai_kelas_tidak_bagus):
    if nilai_kelas_bagus > nilai_kelas_tidak_bagus:
        return "Bagus"
    else:
        return "Tidak Bagus"


def preprocessing(video):
    print('Menjalankan tahap preprocessing...')
    video = cv2.VideoCapture(video)

    fps = video.get(cv2.CAP_PROP_FPS)

    global TOTAL_FRAME
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    TOTAL_FRAME = total_frame

    global DURATION
    DURATION = float(total_frame) / float(fps)

    tempFrame = []
    temp = []

    width = 600
    height = 300
    block_size = 100
    x = int(width / block_size)
    y = int(height / block_size)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilasi = cv2.dilate(thresh1, kernel, iterations=1)
        result = cv2.resize(dilasi, (width, height))

        img = np.uint8([[[0, 0, 0]]])
        for i in range(0, y):
            for j in range(0, x):
                img = cv2.rectangle(result, (0, i * block_size), ((j + 1) * block_size, (i + 1) * block_size),
                                    (0, 0, 0), 1)
                cropped_box = img[i * block_size:(i + 1) * block_size, j * block_size: (j + 1) * block_size]
                tempFrame.append(cropped_box)

        temp.append(img)

    main_process(tempFrame, x, y)


def main_process(tempframe, x, y):
    print('Menjalankan tahap perhitungan nilai POC...')
    data = np.array(tempframe)
    boxFrame = x * y
    idx = boxFrame
    value = 1
    nilai = 0
    totalBox = 0
    tempCor = []
    totalnilainol = 0

    for i in range(0, data.shape[0]):
        if idx < data.shape[0]:
            frameA = data[i]
            frameB = data[idx]
            corImage = poc_process(frameA, frameB)

            tempCor.append(corImage)

            if corImage != 1:
                nilai += corImage
                totalBox += 1
                if corImage == 0:
                    totalnilainol += 1

        idx += 1
        value += 1

    tempCor = np.array(tempCor)
    tempCor = tempCor.reshape((-1, boxFrame))
    data = []

    for i in range(0, tempCor.shape[0]):
        idx = []
        for j in range(0, tempCor.shape[1]):
            if tempCor[i, j] != 1:
                idx.append(j)
        if len(idx) > 0:
            for k in range(len(idx)):
                if idx[k] not in data:
                    data.append(idx[k])

    banyak = len(data)
    rerata = round(nilai / totalBox, 5)
    totalnilainol = round((totalnilainol / totalBox) * 100, 3)

    global RERATA
    RERATA = rerata

    global BANYAK_BOX
    BANYAK_BOX = banyak

    global PERSENTASE_NILAI_NOL
    PERSENTASE_NILAI_NOL = totalnilainol

    global LABEL
    LABEL = naive_bayes_process(rerata, banyak, totalnilainol)

    print(f'Rerata : {rerata}, banyak box : {banyak}, persentase nilai nol : {totalnilainol}')


def poc_process(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)
    arr = R[R >= 0]
    r = np.array(arr.min())
    o = r.real
    p = float(o)
    q = round(p, 4)
    return q


if __name__ == "__main__":
    if not db_check_is_exist():
        db_create_table()
        db_import_data_training()

    app.run(host=my_ip)
