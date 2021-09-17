# Laporan Proyek Machine Learning - Fahmi Jabbar

## Domain Proyek
Domain proyek yang dipilih dalam proyek _machine learning_ ini adalah mengenai **kesehatan** dengan judul proyek "Prediksi Data Air yang Dapat Dikonsumsi Manusia".
- Latar Belakang 

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/58651943/133801531-33e0b221-e741-406b-97f5-cbf455ad4bd1.png" alt="Sumber : https://www.un.org/sustainabledevelopment/wp-content/uploads/2019/07/E_Infographic_06.pdf">
</p>

Akses untuk memiliki air minum bersih dan dapat dikonsumsi merupakan sebuah kebutuhan yang penting untuk dipenuhi oleh manusia. Hal ini juga merupakan tujuan yang selaras dan dimiliki oleh setiap negara termasuk organisasi internasional seperti PBB _(Perserikatan Bangsa-Bangsa)_ yang telah menetapkannya dalam SDG _(Sustainable Development Goals)_. Air juga merupakan kebutuhan hidup yang bukan hanya dikonsumsi, tetapi juga digunakan seperti pada kegiatan mencuci, memasak, dsb. Namun berdasarkan fakta dilapangan, tidak semua orang di dunia ini mempunyai kemudahan tersebut. Berdasarkan hasil laporan pemantauan gabungan WHO dan UNICEF, sebanyak 2 miliar orang di dunia mengalami masalah dalam mengakses layanan air minum yang dikelola dengan aman [[1](https://www.nature.com/articles/s41545-020-00085-z#citeas)]. Akibatnya hal ini dapat mendesak manusia untuk meminum air yang memiliki kualitas kurang baik daripada mengorbankan diri untuk dehidrasi. 

Dampaknya meminum air dengan kualitas yang kurang baik dapat menyebabkan berbagai macam penyakit seperti diare, alergi dermatitis, hingga yang lebih parah seperti kerusakan ginjal, kanker, demineralisasi tulang, dsb [[2](https://www.sciencedirect.com/science/article/abs/pii/S2214714419304453)]. Maka dari itu diperlukan kesadaran bagi semua orang, karena masalah ini merupakan kebutuhan dasar yang seharusnya dipenuhi pada setiap orang. Salah satunya pada proyek ini, dimana akan dibuat sebuah model _machine learning_ untuk mengklasifikasikan data air yang dapat dikonsumsi oleh manusia. Dengan adanya model _machine learning_ ini diharapkan dapat memudahkan ahli seperti ahli hidrologi dan para pencari sumber air dalam mencari air dan mengujinya secara cepat sebelum menunggu hasil dari laboratorium. Implementasinya model ini dapat dijalankan pada sebuah aplikasi web ataupun android.

## Business Understanding

### Problem Statements
Berangkat dari latar belakang diatas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini :
- Bagaimana cara melakukan pra-pemrosesan data air agar dapat digunakan untuk membuat model yang baik?
- Bagaimana cara membuat model _machine learning_ untuk mengklasifikasikan data air yang dapat dikonsumsi oleh manusia?

### Goals
Berikut adalah tujuan dari dibuatnya proyek ini :
- Melakukan pra-pemrosesan data air yang baik agar dapat digunakan dalam membuat model.
- Membuat model _machine learning_ untuk mengklasifikasikan data air yang dapat dikonsumsi oleh manusia yang memiliki tingkat akurasi > 75%.

### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
- Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
  - Mengisi data yang kosong dengan nilai rata rata atau **_(mean substitution)_**
  - Mengatasi data yang tidak seimbang jumlahnya dengan label lain menggunakan teknik **_resample_**
  - Melakukan **pembagian dataset** menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji
  - Menghapus data pencilan pada data latih dengan metode LOF **_Local Outlier Factor_**
  - Melakukan **standardisasi data** pada semua fitur data.
  
  Poin pra-pemrosesan data akan dibahas lebih lanjut pada bagian `Data Preparation`.

- Untuk pembuatan model dipilih penggunaan model dengan algoritma **K-Nearest Neighbor**. Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus klasifikasi. Algoritma mengasumsikan bahwa sesuatu yang serupa pasti selalu berdekatan. Cara kerja algoritma ini adalah sebagai berikut (diterjemahkan dari [[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]):
  - Muat datanya
  - Inisialisasi nilai K (banyak tetangga/kelompok)
  - Pada setiap datanya :
    - Hitung euclidian distance antara contoh kueri dan contoh yang ada pada data tersebut dengan rumus seperti berikut ini :
    ![Rumus Euclidian Distance](https://user-images.githubusercontent.com/58651943/133823136-ede96318-8fa8-4e93-a35f-64a66e5b5fd0.png)  
    - Tambahkan jarak dan urutan dari contoh pada koleksi yang berururutan
  - Pilih entri K paling awal pada koleksi yang berurutan
  - Dapatkan label dari dari entri K yang dipilih
  - Apabila kasus regresi, kembalikan nilai rata-ratanya. Apabila kasus klasifikasi, kembalikan labelnya.
  
  Selain itu, berikut ini merupakan kelebihan dan kekurangan algoritma dari K-Nearest Neighbor (diterjemahkan dari [[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]):
  - Kelebihan :
    - Algoritmanya mudah digunakan dan sederhana
    - Algoritmanya sangat fleksibel, dapat diimplementasikan pada kasus klasifikasi, regresi dan pencarian
  - Kekurangan :
    - Algoritme menjadi lebih lambat secara signifikan karena jumlah contoh dan/atau prediktor/variabel yang meningkat.

## Data Understanding
![Sampul Dataset](https://user-images.githubusercontent.com/58651943/133802226-a7f08f75-0cf0-4b6b-8384-678cf09b15dd.png)

Informasi dataset dapat dilihat pada tabel dibawah ini :

Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset : Water Quality](https://www.kaggle.com/adityakadiwal/water-potability)
Lisensi | CC0: Public Domain
Kategori | Lingkungan, Bumi dan Alam, Kesehatan Publik
Rating Penggunaan | 10.0 (Gold)
Jenis dan Ukuran Berkas | CSV (525 kb)

Pada berkas yang diunduh yakni `water_potability.csv` berisi informasi metriks kualitas air untuk 3276 jenis air yang berbeda. Terdapat 9 buah data numerik (tipe data float64) dan 1 buah data kategori (tipe data int64). Terdapat juga beberapa kolom yang memiliki data kosong diantaranya pada kolom `pH`, `Sulfate` dan `Trihalomethanes`. Untuk penjelasan mengenai variabel-variable pada data _water quality_ dapat dilihat pada poin-poin berikut :

1. `pH` Parameter penting dalam mengevaluasi keseimbangan asam-basa air. Pada data ini nilainya berada pada rentang 0-14. Tingkat pH untuk air yang dapat dikonsumsi menurut rekomendasi WHO adalah 6.5 - 8.5

2. `Hardness` Kesadahan air/Air keras didefinisikan sebagai kapasitas air untuk mengendapkan sabun yang disebabkan oleh Kalsium dan Magnesium. Pada kolom ini kapasitas air untuk mengendapkan sabun ditulis dalam satuan mg/L

3. `Solids` Kemampuan air untuk melarutkan berbagai mineral atau garam organik dan anorganik, ditulis dalam satuan ppm. Untuk keperluan air minum batasnya adalah antara 500-1000 mg/L

4. `Chloramines` Kandungan Klorin/kloramin yang merupakan disinfektan utama yang digunakan dalam sistem air publik, ditulis dalam satuan ppm. Kadar klorin yang baik untuk air minum adalah kurang dari 4 ppm

5. `Sulfate` Kandungan sulfat yang merupakan zat alami yang ditemukan di mineral, tanah, dan batuan. Data ini ditulis dalam satuan mg/L. Kadar sulfat yang ada pada air bersih berkisar antara 3-30 mg/L

6. `Conductivity` Kemampuan air dalam menghantarkan listrik, yang ditulis dalam satuan μS/cm. Pada air minum, nilai konduktivitas air harus berada dibawah 400 μS/cm

7. `Organic_carbon` Kandungan karbon dalam senyawa organik, ditulis dalam satuan ppm. Menurut US EPA kandungan yang kurang dari 2 mg/L dapat digunakan sebagai air olahan/minum

8. `Trihalomethanes` Kandungan Trihalomethanes (THM) yang biasa ditemukan pada air yang diolah dengan klorin. Kadar THM hingga 80 ppm dianggap aman untuk digunakan sebagai air minum

9. `Turbidity` Tingkat kekeruhan air yang diukur dengan tingkat pancaran cahaya pada air dalam satuan NTU. Tingkat kekeruhan kurang dari 5.00 NTU dapat digunakan sebagai air minum menurut WHO

10. `Potability` Menentukan apakah air dapat diminum (nilai 1) atau tidak dapat diminum (nilai 0)

Kemudian terdapat juga visualisasi data untuk kolom dengan fitur numerik (1-9) seperti pada gambar dibawah ini :

![kolom pH](https://user-images.githubusercontent.com/58651943/133829133-fb761045-e0a0-42dc-9de4-7d585d76d493.png)

![kolom Hardness](https://user-images.githubusercontent.com/58651943/133829172-367794f5-75b9-4c4d-8ba9-efc5a1462aba.png)

![kolom Solids](https://user-images.githubusercontent.com/58651943/133829499-2c411629-1038-45fc-a44a-a34fa587fbc3.png)

![kolom Chloramine](https://user-images.githubusercontent.com/58651943/133829556-2443ef9a-e5db-4a79-8bda-1cefa254dfab.png)

![kolom Sulfate](https://user-images.githubusercontent.com/58651943/133829618-6f7734d3-8b96-4e63-baf4-53df19e0508b.png)

![kolom Conductivity](https://user-images.githubusercontent.com/58651943/133829723-6ac4c2db-9b3d-40aa-b8ff-a53f10343ec0.png)

![kolom Organic_carbon](https://user-images.githubusercontent.com/58651943/133829773-4bf0d146-8621-423b-8c56-632eb6614df0.png)

![kolom Trihalomethanes](https://user-images.githubusercontent.com/58651943/133829922-7913210d-c9d4-4b77-a332-d91f78bd7ff3.png)

![kolom Turbidity](https://user-images.githubusercontent.com/58651943/133830012-986d15b9-f169-42be-b0a8-1b056abf062f.png)

Terakhir visualisasi data untuk kolom dengan fitur kategori (10) seperti pada gambar dibawah ini :

![kolom Potability](https://user-images.githubusercontent.com/58651943/133825715-0ea5045b-6f76-49f3-ba59-1f9061324157.png)

## Data Preparation
Seperti yang sudah disebutkan sebelumnya pada bagian _Solution statements_, berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :
- Mengisi data yang kosong dengan nilai rata rata atau **_(mean substitution)_**

  ![Data NaN](https://user-images.githubusercontent.com/58651943/133825321-37e59d59-b895-4787-ba7e-0a9531cf224a.png)
  
  Karena data yang kosong pada dataset cukup banyak, pemilihan metode untuk menghapus data saja bukanlah hal yang bijak. Hal tersebut akan mengakibatkan model yang nantinya akan dibuat kehilangan banyak informasi. Sehingga dipilihlah cara untuk memanipulasi datanya, dengan mengisi data yang kosong dengan nilai rata-rata kolomnya. Data rata-rata kolom dipilih karena merupakan data yang dipastikan bukan data pencilan. Sehingga dengan menganggap data kosong sebagai data rata-rata, model tetap dapat memperoleh informasi dari data yang ada pada kolom lainnya.
  Proses yang dilakukan pertama-tama dengan cara mengambil nilai rata-rata dari kolom yang memiliki data kosong, kemudian memasukannya kepada setiap data kosong sebagai pengganti dari datanya. Semua proses tersebut dilakukan dengan slicing data dengan kondisi menggunakan pandas.
  
- Mengatasi data yang tidak seimbang jumlahnya dengan label lain menggunakan teknik **_resample_**

  Dataset yang tidak seimbang pada data kategori akan menyebabkan model yang dibuat menjadi _bias_ terhadap suatu kategori yang memiliki data lebih banyak. Oleh karena itu diperlukan teknik manipulasi data, dan yang digunakan di sini adalah teknik _resample_. Prosesnya adalah dengan memasukan kolom yang memiliki data paling sedikit pada fungsi [_resample_](https://scikit-learn.org/0.24/modules/generated/sklearn.utils.resample.html#sklearn.utils.resample), kemudian fungsi _resample_ akan menghasilkan data baru dari data yang sudah ada sebelumnya sampai jumlah datanya sama dengan data mayoritas dari label selainnya. Setelah itu selesai, masukan datanya kedalam dataset agar menjadi satu kesatuan data.
  
- Melakukan **pembagian dataset** menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji

  Agar dapat menguji performa model pada data sebenarnya, maka perlu dilakukan pembagian dataset kedalam dua atau tiga bagian. Pada proyek ini dilakukan dua bagian saja yakni pada data latih dan data uji dengan rasio 80:20. Data latih dilakukan sepenuhnya untuk melatih model, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan diharapkan model dapat memiliki performa yang sama baiknya pada data uji seperti pada data latih. Pada bagian ini dipastikan juga pembagian label kategorikal haruslah sama banyak pada data latih dan data uji. Pembagian dataset dilakukan dengan modul [train_test_split](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) dari scikit-learn.
  
- Menghapus data pencilan pada data latih dengan metode LOF **_Local Outlier Factor_**
  
  Data pencilan merupakan nilai yang tidak normal pada dataset dan dapat mengakibatkan distorsi pada analisis statistika dan berujung pada pembuatan model yang kurang optimal [[4](https://statisticsbyjim.com/basics/remove-outliers/)]. Maka dari itu, pada bagian ini diterapkan metode Local Outlier Factor untuk mendeteksi nilai outlier dan kemudian menghapusnya dari data latih. Mengapa data latih saja? Agar kita dapat melihat bagaimana performa model pada data yang belum pernah dilihat model sebelumnya termasuk juga data pencilan. Mengapa dipilih metode LOF? karena metode ini berhubungan erat prosesnya dengan algoritma _nearest neighbor_.
  Mengutip dari dokumentasi [LocalOutlierFactor](https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor), fungsi tersebut bekerja dengan cara menganalisis nilai lokalitas yang ada pada k-tetangga terdekat, yang jaraknya digunakan untuk memperkirakan kepadatan lokal. Dengan membandingkan kepadatan lokal sampel dengan kepadatan lokal tetangganya, seseorang dapat mengidentifikasi sampel yang memiliki kepadatan jauh lebih rendah daripada tetangganya. Apabila kepadatannya rendah maka ini dianggap outlier.
  
- Melakukan **standardisasi data** pada semua fitur data.

  Tahap terakhir dengan melakukan standarisasi data. Hal ini akan membuat semua fitur numerik berada dalam skala data yang sama juga membuat komputasi dari pembuatan model dapat berjalan lebih cepat karena rentang datanya hanya antara 0-1. Untuk melakukan standarisasi data, digunakan fungsi [MinMaxScaler](https://scikit-learn.org/0.24/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) yang perhitungannya kurang lebih seperti rumus dibawah ini 
  ![Rumus MinMaxScaler](https://user-images.githubusercontent.com/58651943/133828773-ee4e17e9-5109-4ac5-96d2-1c07650e6c1f.png)

## Modeling
Setelah melakukan pra-pemrosesan data yang baik pada tahap modeling akan dilakukan dua hal, yakni tahap pembuatan model _baseline_ dan pembuatan model yang dikembangkan.
- Model _baseline_

  Pada tahap ini saya membuat model dasar dengan menggunakan _modul_ scikit-learn yakni [KNeighborsClassifier](https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) tanpa menggunakan parameter tambahan. Lalu melakukan prediksi kepada data ujinya. 
  
- Model yang dikembangkan

  Kemudian setelah melihat kinerja model _baseline_, agar dapat bekerja lebih optimal lagi maka digunakan sebuah fungsi untuk mencari _hyperparameter_ yang optimal dengan [HalvingGridSearchCV](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV). Setelah ditemukan yang optimal, kemudian _hyperparameter_ tersebut diterapkan ke model _baseline_.
  
Hasilnya dapat dilihat seperti pada tabel berikut ini :

![performa model](https://user-images.githubusercontent.com/58651943/133832133-83717305-7a38-4c33-b206-91fa519d0e19.png)

Pada model _baseline_ nilai akurasinya cukup buruk. Begitupun nilai _f1-score_, _recall_ dan _precision_ pada setiap labelnya. Namun setelah dilakukan pengaturan _hyperparameter_, nilai akurasi pun meningkat. Begitupun nilai _f1-score_, _recall_ dan _precision_ pada setiap labelnya. Untuk membuktikannya, kedua model tersebut diuji pada data uji dan di visualisasikan pada _confussion matrix_ seperti berikut.

- Model _baseline_

![performa model baseline](https://user-images.githubusercontent.com/58651943/133832795-a6cc120e-1153-42d3-967f-a0a22df4c9e3.png)


- Model yang dikembangkan

![performa model improvement](https://user-images.githubusercontent.com/58651943/133833185-32e102f2-6d81-4f00-b06b-b8f3c7313903.png)

Dengan hasil diatas, maka model yang dikembangkan merupakan model yang dipilih untuk digunakan.

## Evaluation
Pada proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan metriks akurasi, _f1-score_, _recall_ dan _precision_. Pada gambar dibawah ini ditampilkan kembali hasil pengukuran model yang dikembangkan dengan metriks akurasi, _f1-score_, _recall_ dan _precision_.

![performa model improvement eval](https://user-images.githubusercontent.com/58651943/133834417-3d8e57b8-9546-4dc9-b5b7-7f58a1abc4ea.png)

- Akurasi

![formula akurasi sklearn](https://user-images.githubusercontent.com/58651943/133834677-91c885d0-a443-4567-b75b-30f106ac8124.png)

Akurasi merupakan metrik untuk menghitung nilai ketepatan model dalam memprediksi data dengan data yang sebenarnya. Akurasi dapat dihitung dengan rumus diatas. Kelebihan dari metriks ini adalah sering digunakan dalam kasus pembuatan model klasifikasi baik itu klasifikasi dua kelas, atau kategori. Kekurangan dari metrik ini adalah dapat bersifat 'menyesatkan' pada data yang tidak seimbang.

 - _precision_

_Precision_ merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik model memprediksi label positif terhadap semua prediksi model berlabel positif. Lalu bagaimana cara menghitungnya, pertama-tama kita perlu mengenali dulu istilah TP,TN,FP,FN. Penjelasan singkatnya dapat dilihat pada tabel dibawah ini

![tp,tn,fp,fn](https://user-images.githubusercontent.com/58651943/133837008-ce49e685-d592-475e-b6b9-00e007123a47.png)

Setelah memahaminya, kitapun dapat menghitungnya dengan rumus dibawah ini

![formula precision sklearn](https://user-images.githubusercontent.com/58651943/133837478-fe8bb36a-8964-4133-8cad-d7ad308e6bff.png)

Kelebihan dari metriks ini berfokus pada bagaimana performa (prediksi) model terhadap label data positif, kekurangannya metriks ini tidak memperhitungkan label negatifnya.

- _Recall_

_Recall_ merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik model memprediksi label positif terhadap semua label data positif. Cara menghitungnya dapat dilihat pada rumus dibawah ini 

![formula recall sklearn](https://user-images.githubusercontent.com/58651943/133840605-edcd7b7e-2b82-44fe-8fd8-7acecf754c55.png)

Kelebihan dari metriks ini menghitung bagian negatif dari prediksi label positif (tidak seperti precision). Tetapi kekurangannya ketika semua prediksi = 1 maka _recall_ akan bernilai 1 (tidak memperhitungkan prediksi negatif).

- _f1-score_

_f1-score_ merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik hasil prediksi model (_precision_) dan seberapa lengkap hasil prediksinya (_recall_). Cara menghitungnya dapat dilihat pada rumus dibawah ini 

![formula f1-score sklearn](https://user-images.githubusercontent.com/58651943/133841853-6482710c-b233-4697-8bd7-2bb709e27eaf.png)

Catatan : Nilai beta = 1 (f1-score)

Kelebihan dari metriks ini menutup semua kekurangan yang ada pada _precision_ dan _recall_. Namun kekurangannya adalah _f1-score_ tidak memperhitungkan hasil prediksi benar pada label negatif.


## _Referensi:_

[[1](https://www.nature.com/articles/s41545-020-00085-z)] Bain, R., Johnston, R. & Slaymaker, T. _Drinking water quality and the SDGs._ npj Clean Water 3, 37 (2020). https://doi.org/10.1038/s41545-020-00085-z

[[2](https://www.sciencedirect.com/science/article/abs/pii/S2214714419304453)] Hasan, H. A., & Muhammad, M. H. (2020). _A review of biological drinking water treatment technologies for contaminants removal from polluted water resources._ Journal of Water Process Engineering, 33, 101035. https://doi.org/10.1016/j.jwpe.2019.101035

[[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)] Harrison, O. (2019, July 14). _Machine Learning Basics with the K-Nearest Neighbors Algorithm_. Medium. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

[[4](https://statisticsbyjim.com/basics/remove-outliers/)] Frost, J. (2021, April 5). _Guidelines for Removing and Handling Outliers in Data._ Statistics By Jim. https://statisticsbyjim.com/basics/remove-outliers/

**---Ini adalah bagian akhir laporan---**