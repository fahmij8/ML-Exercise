# Laporan Proyek Machine Learning - Fahmi Jabbar

## Daftar Isi

-   [Project Overview](#project-overview)
-   [Business Understanding](#business-understanding)
-   [Data Understanding](#data-understanding)
-   [Data Preparation](#data-preparation)
-   [Modeling](#modeling)
-   [Evaluation](#evaluation)
-   [Referensi](#referensi)

## Project Overview

<p align="center">
  <img width="500" height="250" src="https://user-images.githubusercontent.com/58651943/134537916-c5d87bdc-1734-4bc2-9e7f-77315c45a8d4.png" alt="Sumber : https://www.kaggle.com/mehdislim01/google-play-store-apps-reviews-110k-comment">
</p>

Pada proyek ini, akan dibuat sebuah sistem rekomendasi aplikasi untuk pengguna di Google Play Store. Google Play Store merupakan platform penyedia aplikasi terbesar dan paling terkenal di dunia, khususnya untuk pengguna android [[1](https://irjet.com/archives/V8/i4/IRJET-V8I4745.pdf)]. Setiap harinya platform ini dibanjiri oleh munculnya banyak aplikasi baru yang dibuat oleh para pengembang aplikasi android [[1](https://irjet.com/archives/V8/i4/IRJET-V8I4745.pdf)]. Tentunya hal tersebut memiliki dampak positif dan dampak negatif. Dampak positifnya adalah munculnya banyak pengembang aplikasi baru yang mulai merilis aplikasinya di Google Play Store, kemudian banyaknya jenis-jenis aplikasi yang dapat dipilih pengguna agar sesuai dengan kebutuhannya.

Di sisi lain, munculnya banyak aplikasi ini membuat pengguna kebingungan untuk memilih aplikasi mana yang perlu diunduh untuk memenuhi kebutuhannya. Sehingga perilaku pengguna dalam mengunduh aplikasi sering mengikuti tren yang sedang ada atau sering juga disebut dengan efek bandwagon [[2](https://journals.sagepub.com/doi/10.1155/2015/475163)]. Misalnya pada sebuah aplikasi komunikasi, dulu kita sering menggunakan LINE. Kemudian muncul beberapa pesaingnya seperti Kakao Talk, WeChat, dll. Namun seiring waktu banyak pengguna yang mulai beralih ke WhatsApp daripada beberapa aplikasi yang sebelumnya disebutkan. Sehingga applikasi WhatsApp pun lebih dipilih sebagai aplikasi utama untuk komunikasi.

Bagi pengembang, hal tersebut merupakan sebuah kerugian karena aplikasinya sudah mulai ditinggalkan pengguna walaupun pada aplikasi yang dibuatnya sudah memiliki fitur yang tidak ada di aplikasi yang sedang digunakan banyak orang. Oleh karena itu diperlukan sebuah sistem rekomendasi agar hal-hal seperti ini tidak terjadi. Selain sebagai fungsi periklanan, sistem rekomendasi juga membuat aplikasi-aplikasi yang jarang diunduh pengguna menjadi kembali dikenal karena sebelumnya sulit untuk ditemukan atau bahkan mempermudah pengguna menemukan aplikasi yang diharapkan.

[← Kembali ke Daftar Isi](#daftar-isi)

## Business Understanding

### Problem Statements

Setelah mengetahui beberapa masalah diatas, berikut ini merupakan rincian masalah yang perlu diselesaikan di proyek ini:

-   Sistem rekomendasi apa yang baik untuk diterapkan pada kasus ini?
-   Bagaimana cara membuat sistem rekomendasi aplikasi untuk pengguna di Google Play Store?

### Goals

Berikut adalah tujuan dari dibuatnya proyek ini:

-   Membuat sistem rekomendasi aplikasi untuk pengguna di Google Play Store.
-   Memberikan rekomendasi untuk aplikasi yang kemungkinan disukai pengguna.

### Solution approach

Gambar dibawah ini adalah diagram alir langkah-langkah yang dilakukan untuk melaksanakan proyek ini :

![Diagram Alir](https://user-images.githubusercontent.com/58651943/134550445-a2595ae1-89f7-439e-ac2f-b6a8d656ebc2.png)

Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :

-   Untuk bagian pra-pemrosesan data dilakukan beberapa teknik diantaranya :

    -   Memperbaiki tipe data pada setiap kolom.
    -   Mengimputasi data kosong pada kolom rating.
    -   Membersihkan data kosong pada kolom.
    -   Membersihkan data duplikasi.

    Setelah hal tersebut dilakukan, selanjutnya dilakukan visualisasi data yang dapat dilihat lebih lengkap pada bagian _Data Understanding_.

-   Untuk bagian persiapan data (sebelum dimasukkan ke model) dilakukan beberapa teknik diantaranya :

    -   Konversi label kategori menjadi _one-hot encoding_.
    -   Standarisasi label numerik.

    Penjelasan lengkap mengenai persiapan data dapat dilihat lebih lengkap pada bagian _Data Preparation_.

-   Kemudian untuk sistem rekomendasi yang dibuat, dipilih sistem rekomendasi _content based filtering_ karena sesuai dengan datasetnya. Sehingga sistem rekomendasi dibuat untuk memberikan rekomendasi pada pengguna terhadap aplikasi yang sebelumnya disukai/diunduh. Beberapa algoritma yang digunakan untuk membuat sistem rekomendasi di proyek ini diantaranya :

    -   Dengan model, yakni algoritma K-Nearest Neighbor. Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus clustering di sistem rekomendasi. Algoritma ini mengasumsikan bahwa sesuatu yang serupa pasti selalu berdekatan. Cara kerja algoritma ini adalah sebagai berikut (diterjemahkan dari [[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]):

        -   Muat datanya
        -   Inisialisasi nilai K (banyak tetangga/kelompok)
        -   Pada setiap datanya :
            -   Hitung euclidian distance antara contoh kueri dan contoh yang ada pada data tersebut dengan rumus seperti berikut ini :
                ![Rumus Euclidian Distance](https://user-images.githubusercontent.com/58651943/133823136-ede96318-8fa8-4e93-a35f-64a66e5b5fd0.png)
            -   Tambahkan jarak dan urutan dari contoh pada koleksi yang berururutan
        -   Pilih entri K paling awal pada koleksi yang berurutan
        -   Dapatkan label dari dari entri K yang dipilih
        -   Apabila kasus regresi, kembalikan nilai rata-ratanya. Apabila kasus klasifikasi, kembalikan labelnya.

        Selain itu, berikut ini merupakan kelebihan dan kekurangan algoritma dari K-Nearest Neighbor (diterjemahkan dari [[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]):

        -   Kelebihan :
            -   Algoritmanya mudah digunakan dan sederhana
            -   Algoritmanya sangat fleksibel, dapat diimplementasikan pada kasus klasifikasi, regresi dan pencarian
        -   Kekurangan :
            -   Algoritme menjadi lebih lambat secara signifikan karena jumlah contoh dan/atau prediktor/variabel yang meningkat.

    -   Dengan _cosine similarity_. Algoritma ini dipilih karena mudah digunakan dan juga sebagai pembanding dengan sistem rekomendasi dengan model. _Cosine similarity_ singkatnya digunakan untuk mengukur kemiripan antara dua buah vektor dan kesamaan arahnya dengan cara menghitung sudut kosinus dari kedua vektornya. Cara menghitungnya adalah dengan rumus berikut ini :

        ![Rumus Cosine Similarity](https://user-images.githubusercontent.com/58651943/134554771-8f23cc13-ef84-4afa-b614-816b6daf65f6.png)

        Dimana nilai x, y adalah nilai vektor dan k adalah nilai _cosine similarity_ dari vektor x dan y.

[← Kembali ke Daftar Isi](#daftar-isi)

## Data Understanding

![Sampul Dataset](https://user-images.githubusercontent.com/58651943/134548111-e092e12e-7621-4b61-86c5-3bc32bcbbfeb.png)

Tabel dibawah ini merupakan informasi dari dataset yang digunakan :

| Jenis                   | Keterangan                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| Sumber                  | [Kaggle Dataset : Google Play Store Apps](https://www.kaggle.com/adityakadiwal/water-potability) |
| Lisensi                 | Creative Commons Attribution 3.0                                                                 |
| Kategori                | Bisnis, Internet, Sains Komputer                                                                 |
| Rating Penggunaan       | 7.1 (Gold)                                                                                       |
| Jenis dan Ukuran Berkas | zip (9 MB)                                                                                       |

Gambar dibawah ini merupakan pratinjau dari dataset pada berkas <code>googleplaystore.csv</code> :

![Pratinjau googleplaystore.csv](https://user-images.githubusercontent.com/58651943/134555896-96029376-fd93-4e90-ace0-24d028e5737d.png)

Kemudian gambar dibawah ini adalah informasi dataset pada berkas <code>googleplaystore.csv</code> :

![Informasi googleplaystore.csv](https://user-images.githubusercontent.com/58651943/134556179-d37b8d77-e8fe-4756-8dda-2f2922721f9c.png)

Berkas <code>googleplaystore.csv</code> berisi informasi mengenai detail aplikasi yang ada di Google Play Store. Namun pada datanya masih terdapat banyak sekali nilai kosong seperti pada kolom <code>Rating</code>, <code>Type</code>, <code>Content Rating</code>, <code>Current Ver</code> dan <code>Android Ver</code>. Berikut ini adalah uraian variabel dari setiap kolom pada dataset :

1. Kolom <code>App</code> merupakan kolom dengan data nama dari aplikasi.

2. Kolom <code>Category</code> merupakan kolom dengan data jenis kategori dari aplikasi.

3. Kolom <code>Rating</code> merupakan kolom dengan data penilaian pengguna dari aplikasi dalam satuan bintang.

4. Kolom <code>Reviews</code> merupakan kolom dengan data jumlah pengguna yang telah memberi ulasan pada aplikasi

5. Kolom <code>Size</code> merupakan kolom dengan data ukuran dari aplikasi dalam satuan byte.

6. Kolom <code>Installs</code> merupakan kolom dengan data jumlah pengguna yang telah mengunduh dan memasang aplikasi.

7. Kolom <code>Type</code> merupakan kolom dengan data jenis aplikasi yang hanya berisi 2 kategori yakni <i>Paid</i>/berbayar dan <i>Free</i>/gratis.

8. Kolom <code>Price</code> merupakan kolom dengan data harga dari aplikasi dalam satuan dollar.

9. Kolom <code>Content Rating</code> merupakan kolom dengan data kategori usia penggunaan untuk aplikasi, seperti <i>children</i>/<i>adult</i> dsb.

10. Kolom <code>Genres</code> merupakan kolom dengan data kategori dari genre aplikasi. Setiap aplikasi bisa saja memiliki dua genre yang berbeda.

11. Kolom <code>Last Updated</code> merupakan kolom dengan data tanggal terakhir aplikasi di perbaharui oleh pengembang.

12. Kolom <code>Current Ver</code> merupakan kolom dengan data versi terkini aplikasi.

13. Kolom <code>Android Ver</code> merupakan kolom dengan data versi android (minimal) yang dibutuhkan untuk memasang aplikasi.

Terakhir, kumpulan gambar dibawah ini merupakan visualisasi data dari dataset yang digunakan (Catatan : Data divisualisasikan setelah dibersihkan) :

-   Data Numerik

![Visualisasi Rating + Type](https://user-images.githubusercontent.com/58651943/134556654-5d9f3a45-2455-4237-950f-fc06ebfef2c1.png)

![Visualisasi Size + Type](https://user-images.githubusercontent.com/58651943/134556793-147ceba8-421d-41cb-accf-65603dfcd3d0.png)

![Visualisasi Reviews + Category + Type](https://user-images.githubusercontent.com/58651943/134556920-e60fa8cb-0e95-47e7-a166-7c1ad957f78d.png)

![Visualisasi Price + Category + Type](https://user-images.githubusercontent.com/58651943/134557011-0aca2aaf-ac21-45ad-bca4-a1909dfbf43f.png)

-   Data Kategori

![Visualisasi Category + Content Rating + Type + Installs](https://user-images.githubusercontent.com/58651943/134557218-85a29848-d311-47dd-b2d5-acff07c70ced.png)

![Visualisasi Genres + Content Rating + Type + Installs](https://user-images.githubusercontent.com/58651943/134557401-16df0652-7f21-4ea7-9bb4-808c12e57492.png)

[← Kembali ke Daftar Isi](#daftar-isi)

## Data Preparation

Seperti yang sudah dijelaskan pada bagian _Solution approach_, berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :

-   Memperbaiki tipe data pada setiap kolom. Hal ini dilakukan karena tipe data pada setiap kolom belum sesuai dengan data di kolomnya, selain itu juga memperbaiki data yang ada di kolomnya. Proses yang dilakukan pada setiap kolom berbeda-beda, berikut ini merupakan rinciannya :
    -   Kolom Rating : Menghapus data rating yang nilainya > 5, mengimputasi data kosong.
    -   Kolom Reviews : Mengganti tipe data kolom menjadi int.
    -   Kolom Size : Mengganti format _Size_ menjadi dalam format MB.
    -   Kolom Installs : Menghapus simbol "+" dan ",".
    -   Kolom Price : Menghapus simbol "$".
    -   Kolom Last Updated : Mengganti tipe data menjadi _datetime_.
    -   Kolom Type, Android Ver, Current Ver : Menghapus data kosong.
    Dengan memperbaiki data pada kolomnya, tipe data kolom pun secara otomatis berubah.
-   Mengimputasi data kosong pada kolom Rating. Hal ini dilakukan karena banyak sekali data rating yang kosong dan apabila dihapus saja hal ini akan mengakibatkan model yang dibuat kehilangan banyak informasi untuk membangun sistem rekomendasi yang baik. Selain itu terdapat peluang juga untuk mengimputasi data Rating karena datanya memiliki tipe data _float_ dengan rentang 1-5. Proses yang dilakukan untuk mengimputasi data adalah dengan menggunakan [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) dari sklearn. Dikutip dari dokumentasinya, KNNImputer mengimputasi data menggunakan K-Nearest Neighbor. Dengan demikian data Rating akan tetap terjaga distribusi datanya.
-   Membersihkan data kosong pada kolom Type, Android Ver dan Current Ver. Hal ini dilakukan karena data pada kategori ini tidak bisa diimputasi. Apabila diimputasi, akan menyebabkan disinformasi kepada pengguna, misalnya pada kolom Type yang berisi mengenai informasi apakah aplikasi tersebut berbayar atau tidak. Selain itu data yang kosong pada kolom ini hanya sedikit, sehingga tidak banyak informasi yang hilang dari keseluruhan datanya.
-   Membersihkan data duplikasi. Hal ini dilakukan karena dapat menyebabkan munculnya data yang sama sebanyak 2 kali atau lebih pada sistem rekomendasi yang nantinya dibuat. Oleh karena itu data duplikasi ini perlu dihilangkan karena sebenarnya data tersebut sudah ada pada dataset. Proses ini dilakukan dengan menggunakan _method_ `drop_duplicates` pada _dataframe_ dari dataset.
-   Konversi label kategori menjadi _one-hot encoding_. Hal ini dilakukan untuk memudahkan pencarian nilai terdekat dari setiap aplikasi. Sebelumnya data ini merupakan data kategori dan dirubah menjadi data numerik yang ada pada setiap kolom yang berbeda kategori. Proses ini dialkukan menggunakan _method_ `get_dummies` pada kolom _dataframe_ dari dataset yang selanjutnya datanya disatukan pada _dataframe_.
-   Standarisasi label numerik. Hal ini dilakukan agar rentang nilai pada label numerik hanya antara 0-1 sehingga dapat mempercepat komputasinya. Selain itu standarisasi juga membuat semua label numerik memiliki rentang nilai yang sama. Proses ini dilakukan menggunakan [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) dari sklearn. Secara matematis, data yang akan diskala-kan dihitung dengan rumus berikut ini.

    <img width="270" src="https://user-images.githubusercontent.com/58651943/133828773-ee4e17e9-5109-4ac5-96d2-1c07650e6c1f.png" alt="Rumus MinMaxScaler">

[← Kembali ke Daftar Isi](#daftar-isi)

## Modeling

Setelah dilakukan pra-pemrosesan data, selanjutnya adalah membuat sistem rekomendasi _content based filtering_.

1. Dengan model K-Nearest Neighbor

    Untuk membangun model ini, digunakan fungsi (NearestNeighbor)[https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html] dari sklearn dengan parameter metriksnya yakni euclidian. Kemudian fungsi tersebut di inisiasikan sebagai model yang selanjutnya dilakukan _fitting_ terhadap data yang ada pada _dataframe_. Setelah itu dibuat fungsi `getRecommendedApps_model` untuk memberikan rekomendasi terhadap suatu nama aplikasi yang dimana skenarionya adalah apabila pengguna menyukai atau mengunduh aplikasi tersebut, maka berikan rekomendasi ini sebagai aplikasi yang mungkin disukai. Hasil rekomendasinya adalah seperti berikut :

    ![Rekomendasi KNN](https://user-images.githubusercontent.com/58651943/134567304-0e3b7895-61af-46cc-b82b-58a099c0a968.png)

2. Dengan _cosine similarity_

    Selanjutnya, rekomendasi pun dapat diberikan dengan menghitung _cosine similarity_ dari setiap data di dataset menggunakan fungsi [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari sklearn. Prosesnya adalah dengan memanggil fungsi `cosine_similarity` dengan argumen _dataframe_ sebagai objeknya. Kemudian hasil dari perhitungannya disimpan pada dataframe baru. Untuk tahapan pemberian rekomendasinya, dibuat fungsi `getRecommendedApps_cosine` dimana fungsi tersebut akan memberikan rekomendasi terhadap suatu nama aplikasi dengan skenario yang sama dengan nomor 1.

    Pada fungsi tersebut, akan dilakukan pencarian kolom dari suatu nama aplikasi pada _dataframe_ baru hasil perhitungan _cosine similarity_. Kemudian diurutkan nilainya berdasarkan nilai _cosine similarity_ tertinggi dan juga urutannya. Setiap urutan ke 2 terakhir sampai ke n terakhir merupakan kandidat yang memiliki nilai _cosine similarity_ yang sama maka akan ditampilkan sebagai hasil rekomendasinya. Urutan paling terakhir merupakan nilai _cosine similarity_ dari kolom dengan nama aplikasi yang sama. Untuk lebih jelasnya hasil rekomendasi dapat dilihat seperti berikut ini :

    ![Rekomendasi Cosine Similarity](https://user-images.githubusercontent.com/58651943/134568493-043292d8-9002-4e2e-ac9d-7f534d4595a6.png)

[← Kembali ke Daftar Isi](#daftar-isi)

## Evaluation

Untuk mengukur kinerja model KNN untuk sistem rekomendasi digunakan beberapa metriks diantaranya :

1. Skor Calinski Harabasz

    Skor Calinski Harabasz digunakan untuk menghitung kriteria rasio varian. Metriks ini digunakan pada model clustering seperti yang saat ini sedang digunakan. Skor semakin tinggi ketika kluster padat dan terpisah dengan baik. Dikutip dari laman dokumentasi scikit-learn, Skor ini dihitung dengan formula [[4](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)] :

    ![Formula Calinski Harabasz - 1](https://user-images.githubusercontent.com/58651943/134569725-1f36ec4f-fe26-46f0-bbf2-ca06381c69a8.png)

    ![Formula Calinski Harabasz - 2](https://user-images.githubusercontent.com/58651943/134571888-be63b7ce-bcf7-4564-af28-4af9f12c20fb.png)

    Kelebihan dari metriks ini adalah :

    - Skornya tinggi apabila kluster padat dan terpisah dengan baik, yang mana bergantung pada konsep standar dari sebuah kluster.
    - Skornya cepat untuk dihitung.

    Sedangkan kekurangannya :

    - Metriks ini hanya baik digunakan pada kasus _convex cluster_.

    Penerapannya pada kode adalah dengan menggunakan fungsi [calinski_harabasz_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html) dari sklearn. Fungsi tersebut menerima argumen dari sebuah data yang digunakan untuk membuat model dan labelnya. Berikut ini adalah hasil penerapannya pada model KNN.

    ![Skor Calinski Harabasz](https://user-images.githubusercontent.com/58651943/134569489-c29aa25c-f56b-438c-8eb4-8c2b18769fd0.png)

    Pada model ini, nampaknya kluster masih belum padat dan terpisahkan dengan baik karena nilai skornya masih cukup rendah. Memungkinkan rekomendasi pada beberapa aplikasi masih terdapat rekomendasi yang tidak sesuai dengan aplikasi yang disukai pengguna.

2. Skor Davies Bouldin

    Skor Davies Bouldin digunakan untuk menilai separasi tiap kluster dari model. Metriks ini digunakan pada model clustering seperti yang saat ini sedang digunakan. Skor rendah ketika separasi tiap kluster di model terpisahkan dengan baik. Dikutip dari laman dokumentasi scikit-learn, Skor ini dihitung dengan formula [[4](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)] :

    ![Formula Davies Bouldin - 1](https://user-images.githubusercontent.com/58651943/134571953-0f466793-eb7d-4c69-8a8e-39f430074502.png)

    ![Formula Davies Bouldin - 2](https://user-images.githubusercontent.com/58651943/134572201-f518acd6-eb48-49ca-aea4-96d46f6483cb.png)

    Kelebihan dari metriks ini adalah :

    - Komputasinya lebih mudah daripada Skor Silhouette.
    - Skor yang dihitung hanya jumlah dan fitur yang melekat pada dataset.

    Kekurangan dari metriks ini adalah :

    - Metriks ini hanya baik digunakan pada kasus _convex cluster_.
    - Penggunaan jarak centroid membatasi metriks jarak ke ruang Euclidean

    ![Skor Davies Bouldin](https://user-images.githubusercontent.com/58651943/134569215-57cee0eb-ad46-4e08-a702-b4f7f65ae62f.png)

    Pada model ini skornya cukup kecil sehingga menandakan modelnya sudah memiliki separasi kluster yang baik. Hal ini dibuktikan juga dengan hasil rekomendasi aplikasi yang cukup baik dan sesuai kategorinya.

[← Kembali ke Daftar Isi](#daftar-isi)

# Referensi

[[1](https://irjet.com/archives/V8/i4/IRJET-V8I4745.pdf)] Sunasara, A. A., Jaiswal, N., Poojari, S., & Chaturvedi, A. K. (2021). _Play Store App Analysis_. International Research Journal of Engineering and Technology (IRJET).

[[2](https://journals.sagepub.com/doi/10.1155/2015/475163)] Choi, S.-M., Lee, H., Han, Y.-S., Man, K. L., & Chong, W. K. (2015). _A Recommendation Model Using the Bandwagon Effect for E-Marketing Purposes in IoT_. International Journal of Distributed Sensor Networks. https://doi.org/10.1155/2015/475163

[[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)] Harrison, O. (2019, July 14). _Machine Learning Basics with the K-Nearest Neighbors Algorithm_. Medium. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

[[4](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)] scikit-learn. (2021). Clustering - Performance Evaluation. https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

[← Kembali ke Daftar Isi](#daftar-isi)
