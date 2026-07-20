

<!-- Start of picture text -->
warbs Pa<br>isS@ C ><br>55<br>**<br>TS<br><!-- End of picture text -->



<!-- Start of picture text -->
warbs Pa<br>isS@ C ><br>55<br>**<br>TS<br><!-- End of picture text -->

##### **LEMBAR PERSETUJUAN TESIS** 

#### **OPTIMASI BOBOT MODEL** **_ENSEMBLE_ UNTUK PERAMALAN PENJUALAN MENGGUNAKAN** **_GREY WOLF OPTIMIZER_** 

Telah disetujui untuk disidangkan pada Program Studi Teknik Informatika S-2 Universitas Pamulang Pada tanggal ………………… 

Oleh : Bayu Nurcahyono 231012000006 

Tesis ini telah disetujui untuk diajukan ke Tim Penilai/Penguji oleh: Pembimbing I Pembimbing II 

<u>Dr. Ir. Agung Budi Susanto, MM Dr. Sudarno Wiharjo, D.E.A</u> NIDK 8811620016 NIDK 8855500016 

Mengetahui: Kaprodi Teknik Informatika S-2 

<u>Dr. Sajarwo Anggai, S.ST., M.T.</u> NIDN. 0 <mark>421108703</mark> 

i 

UNIVERSITAS PAMULANG 

##### **LEMBAR PENGESAHAN TESIS** 

#### **OPTIMASI BOBOT MODEL** **_ENSEMBLE_ UNTUK PERAMALAN PENJUALAN MENGGUNAKAN** **_GREY WOLF OPTIMIZER_** 

Telah dipertahankan di hadapan Dewan Penguji Program Pascasarjana Universitas Pamulang Pada tanggal ………………… Oleh : _Bayu Nurcahyono 231012000006_ Pembimbing I Pembimbing II 

<u>Dr. Ir. Agung Budi Susanto, MM Dr. Sudarno, DEA</u> NIDK 8811620016 NIDK 8855500016 Penguji I Penguji II <u>………………………… …………………………</u> NIDN ……………… NIDN ……………… Disahkan: Direktur Program Pascasarjana Universitas Pamulang <u>Dr. Saiful Anwar, S.Pd., S.E., M.Pd.</u> NIDN.  0426048503 

ii 

UNIVERSITAS PAMULANG 

##### **PERNYATAAN KEASLIAN TESIS** 

Nama : Bayu Nurcahyono NIM : 23102000006 Program Studi : Teknik Informatika S-2 Judul Tesis : Optimasi Bobot Model Ensemble untuk Peramalan Penjualan Menggunakan Grey Wolf Optimizer 

Dengan ini saya menyatakan bahawa dalam tesis ini tidak terdapat karya yang pernah diajukan untuk memperoleh gelar kesarjanaan di suatu Perguruan Tinggi, dan sepanjang pengetahuan saya juga tidak terdapat karya atau pendapat yang pernah ditulis atau di terbitkan oleh orang lain, kecuali yang secara tertulis diacu dalam naskah tesis ini dan disebutkan dalam daftar pustaka. 

Tangerang Selatan, ….. Juli 2026 

(Bayu Nurcahyono) 

iii 

UNIVERSITAS PAMULANG 

##### **KATA PENGANTAR** 

Puji syukur Alhamdulillah kehadirat Allah SWT yang telah melimpahkan segala rahmat dan karunia-Nya, sehingga penulis dapat menyelesaikan Tesis yang merupakan salah satu persyaratan untuk menyelesaikan program studi strata dua (S2) pada program studi Teknik Informatika di Universitas Pamulang. 

Penulis menyadari skripsi ini masih jauh dari sempurna. Karena itu, kritik dan saran akan senantiasa penulis terima dengan senang hati. Dengan segala keterbatasan, penulis menyadari pula bahwa Tesis ini tidak kan terwujud tanpa bantuan, bimbingan, dan dorongan dari berbagai pihak. 

Dalam kesempataan ini, penulis ingin mengucapkan terima kasih kepada semua pihak yang telah membimbing dan memberikan masukkan berupa kritik serta saran kepada penulis dalam penyusunan tugas akhir ini, oleh karena itu, penulis menyampaikan rasa hormat dan ungkapan terima kasih kepada: 

1. Universitas Pamulang dan Program Studi Teknik Informatika S-2 yang telah melayani proses Akademik dan pembelajaran dengan baik dari mulai pendaftaran mahasiswa baru, pelaksanaan perkuliahan serta sampai penyusunan tugas akhir. 

2. Rektor Universitas Pamulang yang mengijinkan penulis untuk menempuh studi program S-2. 

3. Dr. Saiful Anwar, S.Pd., S.E., M.Pd., selaku Direktur Pasca sarjana Universitas Pamulang. 

4. Dr. Sajarwo Anggai, S.ST., M.T., sebagai Ketua Program Studi Teknik Informatika S-2 Universitas Pamulang. 

5. Dr. Ir. Agung Budi Susanto, MM. dan Dr. Sudarno, DEA, selaku dosen pembimbing I dan Pembimbing II yang telah menyediakan waktu, tenaga dan pikiran untuk mengarahkan penulisan dan penyusunan tesis ini. 

6. Rekan-rekan mahasiswa Program Studi Teknik Informatika S-2 Universitas Pamulang yang telah banyak mendukung penulis dalam menyelesaikan tesis ini. 

7. Semua Pihak yang terlibat dan tidak penulis sebutkan satu persatu. 

iv 

UNIVERSITAS PAMULANG 

Penulis berharap semoga langkah selanjutnya diridhoi oleh Allah SWT. Akhirnya sebagai penutup penulis berharap semoga laporan Tesis ini dapat memberikan manfaat bagi pembaca untuk mengembangkan ilmu pengetahuan, khususnya dibidang teknologi kecerdasan buatan. Aamiin. 

Tangerang Selatan, ….. Juli 2026 

(Bayu Nurcahyono) 

UNIVERSITAS PAMULANG 

v 

##### **PERNYATAAN PERSETUJUAN PUBLIKASI TESIS UNTUK KEPENTINGAN AKADEMIS** 

Sebagai sivitas akademik Universitas Pamulang saya yang bertandatangan di 

bawah ini: 

Nama : Bayu Nurcahyono NIM : 23102000006 Program Studi : Teknik Informatika S-2 Jenis Karya : Tesis 

Demi pengembangan ilmu pengetahuan, menyetujui untuk memberikan kepada Universitas Pamulang Hak Bebas Royalti Noneksklusif ( _Non-exclusive RoyaltyFree Right_ ) atas karya ilmiah saya yang berjudul: 

#### **OPTIMASI BOBOT MODEL** **_ENSEMBLE_ UNTUK PERAMALAN PENJUALAN MENGGUNAKAN** **_GREY WOLF OPTIMIZER_** 

Beserta perangkat yang ada (jika diperlukan). Dengan Hak Bebas Royalti Noneksklusif ini Universitas Pamulang berhak menyimpan, mengalih media/formatkan, mengelola dalam bentuk pangkalan data ( _database_ ), merawat dan mempublikasikan tesis saya selama tetap mencantumkan nama saya sebagai penulis/pencipta dan sebagai pemilik Hak  Cipta. 

Demikian pernyataan ini saya buat dengan sebenarnya. 

Dibuat di : Tangerang Selatan Pada tanggal: ….. Juli 2026 Yang menyatakan 

(Bayu Nurcahyono) 

vi 

UNIVERSITAS PAMULANG 

##### **ABSTRAK** 

Prediksi penjualan memiliki peran krusial dalam mendukung pengambilan keputusan strategis perusahaan, seperti perencanaan produksi, pengelolaan persediaan, dan penyusunan strategi pemasaran. Namun, data penjualan yang bersifat kompleks, non-linear, dan musiman menyebabkan metode konvensional seperti _Moving Average_ (MA), _Exponential Smoothing_ (ES), dan _Recurrent Neural Network_ (RNN) sering menghasilkan akurasi yang terbatas. Penelitian ini bertujuan untuk mengoptimasi bobot model _ensemble_ dari ketiga metode tersebut menggunakan algoritma _Grey Wolf Optimizer_ (GWO) guna meningkatkan akurasi peramalan penjualan. Metode yang diusulkan diawali dengan pra-pemrosesan dataset Store Sales - Time Series Forecasting dari Kaggle, membagi data dengan data latih dan data uji, mengembangkan model baseline individual MA, ES, dan RNN _,_ mengembangkan model _weighted ensemble_ yang terdiri dari ketiga model sebelumnya dengan bobot ( _weight_ ) yang dioptimalkan dengan GWO. Hasil kinerja dari ketiga model _baseline_ individual dan model _weighted ensemble_ dievaluasi menggunakan metrik MAE, MSE, RMSE, MAPE, dan R². Penelitian ini diharapkan menghasilkan model _weighted ensemble_ dengan bobot optimal yang mampu menghasilkan prediksi penjualan lebih akurat dibandingkan model _baseline_ individu. 

Kata kunci: peramalan penjualan _, ensemble learning,_ optimasi _, grey wolf optimizer, machine learning_ 

vii 

UNIVERSITAS PAMULANG 

##### **ABSTRACT** 

_Sales forecasting plays a crucial role in supporting strategic corporate decisionmaking, such as production planning, inventory management, and the formulation of marketing strategies. However, sales data that is complex, non-linear, and seasonal causes conventional methods such as Moving Average (MA), Exponential Smoothing (ES), and Recurrent Neural Network (RNN) to often produce limited accuracy._ 

_This study aims to optimize the ensemble model weights of these three methods using the Grey Wolf Optimizer (GWO) algorithm to improve sales forecasting accuracy. The proposed method begins with preprocessing the Store Sales - Time Series Forecasting dataset from Kaggle, splitting the data into training and testing sets, developing individual baseline models of MA, ES, and RNN, and developing a weighted ensemble model consisting of the three previous models with weights optimized by GWO. The performance of the three individual baseline models and the weighted ensemble model will be evaluated using MAE, MSE, RMSE, MAPE, and R² metrics. This research is expected to produce a weighted ensemble model with optimal weights capable of generating more accurate sales predictions compared to individual baseline models._ 

**_Keywords:_** _sales forecasting, ensemble learning, optimization, grey wolf optimizer, machine learning_ 

viii 

UNIVERSITAS PAMULANG 

#### **DAFTAR ISI** 

|**LEMBAR PERSETUJUAN TESIS ...................................................................... i**|
|---|
|**LEMBAR PENGESAHAN TESIS ...................................................................... ii**|
|**PERNYATAAN KEASLIAN TESIS ................................................................. iii**|
|**KATA PENGANTAR .......................................................................................... iv**|
|**PERNYATAAN PERSETUJUAN PUBLIKASI TESIS UNTUK**|
|**KEPENTINGAN AKADEMIS ........................................................................... vi**|
|**ABSTRAK ........................................................................................................... vii**|
|**ABSTRACT ........................................................................................................ viii**|
|**DAFTAR ISI ......................................................................................................... ix**|
|**BAB I**<br>**PENDAHULUAN ................................................................................1**|
|1.1<br>Latar Belakang .............................................................................1|
|1.2<br>Permasalahan Penelitian ..............................................................3|
|1.2.1<br>Identifikasi Masalah .........................................................................3|
|1.2.2<br>Ruang Lingkup Masalah ..................................................................4|
|1.2.3<br>Rumusan Masalah ............................................................................4|
|1.3<br>Tujuan dan Manfaat Penelitian ....................................................5|
|1.4<br>Sistematika Penulisan ..................................................................6|
|**BAB II**<br>**LANDASAN TEORI DAN KERANGKA PEMIKIRAN ................7**|
|2.1<br>Tinjauan Pustaka ..........................................................................7|
|2.2<br>Landasan Teori ..........................................................................24|
|2.2.1<br>Peramalan (_Forecasting_) ................................................................24|
|2.2.2<br>Peramalan Penjualan (_Sales Forecasting_) ......................................25|
|2.2.3<br>_Time Series Forecasting_.................................................................26|
|2.2.4<br>AIDA (_Attention, Interest, Desire, Action_) ....................................28|
|2.2.5<br>Metode Peramalan_Baseline_...........................................................29|
|2.2.6<br>Metode_Ensemble_...........................................................................32|
|2.2.7<br>Algoritma Optimasi ........................................................................34|
|2.2.8<br>_Grey Wolf Optimizer_......................................................................36|



ix 

UNIVERSITAS PAMULANG 

|2.3|Kerangka Pemikiran ..................................................................44|
|---|---|
|**BAB III**|**METODOLOGI ................................................................................48**|
|3.1|Analisis Kebutuhan ....................................................................48|
|3.1.1|Analisis Kebutuhan Data ................................................................48|
|3.1.2|Analisis Kebutuhan Perangkat Keras .............................................49|
|3.1.3|Analisis Kebutuhan Perangkat Lunak ............................................49|
|3.2|Perancangan Penelitian ..............................................................50|
|3.3|Teknik Analisis ..........................................................................54|
|3.3.1|_Mean Absolute Error_(MAE) .........................................................54|
|3.3.2|_Mean Squared Error_(MSE) ...........................................................54|
|3.3.3|_Root Mean Squared Error_(RMSE) ...............................................55|
|3.3.4|_Mean Absolute Percentage Error_(MAPE) ....................................55|
|3.3.5|_R-squared_(R²) ................................................................................56|
|**BAB IV**|**HASIL DAN PEMBAHASAN .........................................................57**|
|4.1|Hasil ...........................................................................................57|
|4.1.1|Deskripsi Data ................................................................................57|
|4.1.2|Pra Pemrosesan Data ......................................................................59|
|4.1.3|Pembagian Data Latih dan Data Uji ...............................................61|
|4.1.4|Hasil Pengujian Model_Baseline_....................................................62|
|4.1.5|Hasil Optimasi Bobot Ensemble ....................................................67|
|4.2|Pembahasan ...............................................................................70|
|4.2.1|Bobot Ensemble .............................................................................71|
|4.2.2|Perbandingan Performa Model .......................................................71|
|4.2.3|Peningkatan Akurasi Model GWO Ensemble ................................74|
|**BAB V**|**KESIMPULAN DAN SARAN .........................................................75**|
|5.1|Kesimpulan ................................................................................75|
|5.2|Saran ..........................................................................................76|



UNIVERSITAS PAMULANG 

x 

#### **DAFTAR  TABEL** 

|Tabel 2.1 Literatur tentang penelitian sebelumnya .................................................. 7|
|---|
|Tabel  3.1 Data Penjualan ...................................................................................... 49|
|Tabel  3.2 Spesifikasi Perangkat Keras .................................................................. 50|
|Tabel 3.3 Spesifikasi Perangkat Lunak .................................................................. 50|
|Tabel 4.1 Karakteristik Data .................................................................................. 58|
|Tabel 4.2 Analisa Deskriptif Dataset Awal............................................................ 59|
|Tabel 4.3 10 Baris Teratas Dataset Awal ............................................................... 59|
|Tabel 4.4 10 Baris Terbawah Dataset Awal........................................................... 60|
|Tabel 4.5 Analisa Deskriptif Dataset Hasil Pra Pemrosesan ................................. 61|
|Tabel 4.6 10 Baris Teratas Dataset Hasil Pra Pemrosesan..................................... 61|
|Tabel 4.7 10 Baris Terbawah Dataset Hasil Pra Pemrosesan ................................ 62|
|Tabel 4.8 Performa Model_Baseline_....................................................................... 67|
|Tabel 4.9 30-_run_Optimasi GWO .......................................................................... 69|
|Tabel 4.10 Ringkasan Statistik hasil Optimasi GWO ............................................ 70|
|Tabel 4.11 Perbandingan Metrik Performa Model Baseline dan Ensemble .......... 73|



xi 

UNIVERSITAS PAMULANG 

#### **DAFTAR  GAMBAR** 

|Gambar 2.1 Recurrent Neural Network ................................................................. 32|
|---|
|Gambar 2.2 Hirarki sosial serigala abu-abu ........................................................... 37|
|Gambar 2.3 Perilaku berburu serigala abu-abu ...................................................... 38|
|Gambar 2.4 Kemungkinan posisi serigala dalam peta vektor 2D dan 3D ............. 40|
|Gambar 2.5 Perbaruan posisi serigala dalam GWO............................................... 42|
|Gambar 2.6 Menyerang mangsa vs mencari mangsa ............................................. 43|
|Gambar 2.7_Pseudo Code_algoritma GWO ............................................................ 44|
|Gambar 2.8 Diagram Kerangka Pemikiran ............................................................ 46|
|Gambar 3.1 Diagram Perancangan Penelitian ....................................................... 51|
|Gambar 4.1 Visualisasi Pembagian Data Latih dan Data Uji ................................ 63|
|Gambar 4.2 Training Loss Model RNN ................................................................. 64|
|Gambar 4.3 Perbandingan Prediksi Moving Average (MA) dengan Data Uji ...... 65|
|Gambar 4.4 Perbandingan Prediksi Exponential Smoothing (ES) dan Data Uji ... 66|
|Gambar 4.5 Perbandingan Prediksi RNN dengan Data Uji ................................... 67|
|Gambar 4.6 Kurva Konvergensi GWO .................................................................. 71|
|Gambar 4.7 Perbandingan MAPE Seluruh Model ................................................. 73|
|Gambar 4.8 Perbandingan Prediksi GWO Ensemble dengan Data Uji ................. 74|
|Gambar 5.1 Infografis Perbandingan Pendekatan Univariate vs Multivariate ...... 78|



xii 

UNIVERSITAS PAMULANG 

#### **DAFTAR  PERSAMAAN** 

|Persamaan 2.1 Prediksi_Ensemble_.......................................................................... 33|
|---|
|Persamaan 2.2 Vektor Posisi Mangsa - Serigala .................................................... 39|
|Persamaan 2.3 Vektor Posisi Serigala berikutnya.................................................. 39|
|Persamaan 2.4 Vektor koefisien𝐴......................................................................... 39|
|Persamaan 2.5 Vektor koefisien𝐶......................................................................... 39|
|Persamaan 2.7 Posisi serigala_Alpha_...................................................................... 40|
|Persamaan 2.8 Posisi serigala_Beta_........................................................................ 41|
|Persamaan 2.9 Posisi serigala_Delta_...................................................................... 41|
|Persamaan 2.10 Posisi serigala baru ...................................................................... 41|
|Persamaan 3.1_Mean Absolute Error_(MAE) ......................................................... 55|
|Persamaan 3.2_Mean Squared Error_(MSE) .......................................................... 56|
|Persamaan 3.3_Root Mean Squared Error_(RMSE) ............................................... 56|
|Persamaan 3.4_Mean Absolute Percentage Error_(MAPE) ................................... 57|
|Persamaan 3.5_R-Squared_(R<sup>2</sup>) ............................................................................... 57|



xiii 

UNIVERSITAS PAMULANG 

#### **BAB I** 

#### **PENDAHULUAN** 

##### **1.1 Latar Belakang** 

Peramalan ( _forecasting_ ) merupakan salah satu komponen penting dalam proses pengambilan keputusan strategis di berbagai sektor bisnis (Rob J. Hyndman & Athanasopoulos, 2018). Dalam konteks perusahaan, kegiatan peramalan penjualan ( _sales forecasting_ ) berperan sebagai dasar bagi perencanaan produksi, pengelolaan persediaan, pengaturan distribusi, serta penyusunan strategi pemasaran dan keuangan. Akurasi dalam peramalan penjualan sangat menentukan efektivitas kebijakan bisnis (Makridakis et al., 2020), karena hasil prediksi yang tepat memungkinkan perusahaan mengalokasikan sumber daya secara efisien dan menghindari risiko kelebihan maupun kekurangan pasokan produk. 

Namun demikian, karakteristik data penjualan pada umumnya bersifat kompleks, non-linear, dan dinamis. Fluktuasi permintaan sering kali dipengaruhi oleh faktor-faktor musiman, perilaku konsumen, promosi, tren pasar, serta kondisi ekonomi makro. Dalam perspektif pemasaran, kompleksitas pola data penjualan ini dapat dijelaskan melalui model AIDA ( _Attention, Interest, Desire, Action_ ). Model AIDA menggambarkan perjalanan konsumen dari pertama kali menyadari keberadaan produk ( _Attention_ ), kemudian menumbuhkan minat ( _Interest_ ), lalu mengembangkan keinginan untuk membeli ( _Desire_ ), hingga akhirnya melakukan tindakan pembelian ( _Action_ ). Setiap tahap dalam AIDA memberikan kontribusi berbeda terhadap pola penjualan. Misalnya, kegiatan promosi yang terdokumentasi dalam variabel _onpromotion_ pada dataset _Store Sales_ bekerja pada tahap _Attention_ dan _Interest_ . Lonjakan penjualan sering terjadi pada tahap _Action_ , misalnya saat hari libur atau _event_ tertentu. Akumulasi dari perilaku konsumen yang mengikuti siklus AIDA ini menghasilkan data penjualan yang kompleks, non-linear, dan sarat dengan pola musiman, sehingga model peramalan konvensional seperti MA, ES, dan LR seringkali menghasilkan akurasi yang kurang optimal 

Kompleksitas ini menyebabkan model peramalan konvensional seperti _Moving Average_ , _Exponential Smoothing_ , maupun _Autoregressive Integrated Moving Average_ (ARIMA) sering kali menghasilkan akurasi yang kurang 

1 

UNIVERSITAS PAMULANG 

optimal(Aras et al., 2017). Model-model tersebut kurang mengantisipasi hubungan non-linear antar data historis dan tidak cukup adaptif terhadap pola perubahan yang tidak teratur. 

Seiring berkembangnya teknologi analisis data, pendekatan berbasis machine learning mulai banyak digunakan untuk meningkatkan akurasi prediksi penjualan. Metode machine learning memiliki kemampuan untuk mempelajari pola non-linear dan interaksi kompleks antar variabel tanpa memerlukan asumsi distribusi data tertentu(Ganguly & Mukherjee, 2024). Akan tetapi, tidak ada satu algoritma pun yang secara konsisten memberikan hasil terbaik di semua jenis data dan domain. Oleh karena itu, muncul pendekatan _ensemble learning_ , yaitu teknik yang menggabungkan beberapa model prediktif untuk memperoleh hasil yang lebih stabil dan akurat. 

_Ensemble learning_ bertujuan mengurangi kelemahan model tunggal dengan memanfaatkan keunggulan dari setiap model dasar ( _base learner_ ). Dalam penelitian ini, tiga model dasar yang digunakan adalah _Moving Average_ (MA), _Exponential Smoothing_ (ES), dan _Recurrent Neural Network_ (RNN). Ketiga model ini dipilih karena mewakili pendekatan statistik yang umum digunakan dalam peramalan penjualan serta memiliki karakteristik yang saling melengkapi dalam menangani pola data tren dan musiman. Kombinasi ketiganya diharapkan mampu menghasilkan model yang lebih akurat melalui mekanisme _weighted ensemble_ , di mana setiap model diberikan bobot kontribusi berdasarkan kinerjanya terhadap data historis. 

Meskipun _ensemble learning_ telah terbukti efektif, sebagian besar penelitian sebelumnya menggunakan skema bobot tetap ( _fixed weight_ ) atau rata-rata sama ( _equal weight_ ) dalam proses kombinasi model (Adhikari & Agrawal, 2012). Penelitian ini menawarkan kontribusi baru ( _novelty_ ) berupa penerapan algoritma _Grey Wolf Optimizer_ (GWO) untuk menentukan bobot optimal antar model dalam _ensemble learning_ dalam peramalan penjualan. Pendekatan ini memungkinkan proses pembobotan dilakukan secara adaptif dan optimal, bukan berdasarkan aturan tetap. Dengan GWO, bobot model ditentukan melalui proses optimasi yang meniru perilaku berburu serigala abu-abu dalam mencari solusi terbaik. Tujuannya adalah 

UNIVERSITAS PAMULANG 

2 

meminimalkan nilai kesalahan prediksi dengan menyeimbangkan kontribusi setiap model dasar secara dinamis sesuai performanya terhadap data. 

Selain itu, penelitian ini memperluas penerapan konsep _metaheuristic optimization_ pada domain peramalan penjualan, yang selama ini lebih banyak digunakan pada peramalan arus kas (cash flow) atau produksi industri. Penerapan GWO untuk optimasi bobot ensemble masih tergolong baru dan belum banyak dalam literatur _forecasting_ , baik pada konteks keuangan maupun bisnis. Oleh karena itu, penelitian ini diharapkan dapat memberikan kontribusi ilmiah dengan memperkenalkan mekanisme _adaptive weighted ensemble_ berbasis GWO untuk meningkatkan akurasi hasil peramalan penjualan. 

Berdasarkan hal tersebut, penelitian ini mengusulkan pengembangan model _ensemble learning_ yang dioptimasi menggunakan _Grey Wolf Optimizer_ (GWO) untuk meningkatkan akurasi peramalan penjualan. Model ini diharapkan mampu mengatasi keterbatasan metode konvensional dan memberikan hasil peramalan yang lebih akurat, adaptif, serta dapat diandalkan untuk mendukung pengambilan keputusan strategis perusahaan. 

##### **1.2 Permasalahan Penelitian** 

##### **1.2.1 Identifikasi Masalah** 

Berdasarkan latar belakang masalah di atas, identifikasi masalah dalam penelitian ini adalah sebagai berikut: 

- a. Data penjualan umumnya memiliki pola yang tidak linear, musiman, dan dipengaruhi oleh banyak faktor eksternal sehingga sulit dimodelkan menggunakan metode statistik konvensional. 

- b. Model peramalan seperti _Moving Average_ , _Exponential Smoothing_ , dan _Recurrent Neural Network_ mempunyai kelebihan masing-masing, belum belum ada pembuktian apakah jika digabungkan mampu menangkap fluktuasi dan ketidakpastian pola data penjualan yang dinamis. 

- c. Belum banyak penelitian yang mengoptimalkan kombinasi bobot model _ensemble forecasting_ menggunakan algoritma metaheuristik seperti _Grey Wolf Optimizer (GWO)_ 

UNIVERSITAS PAMULANG 

3 

##### **1.2.2 Ruang Lingkup Masalah** 

Berdasarkan beberapa pokok permasalahan yang telah diuraikan pada identifikasi masalah di atas, maka penelitian dibatasi pada: 

- a. Penelitian ini berfokus pada peramalan penjualan ( _sales forecasting_ ) dengan pendekatan regresi deret waktu ( _time series regression_ ) pada data penjualan agregat harian total. Dengan demikian permasalahan yang diselesaikan bersifat _univariate time series,_ sehingga analisis dan pemodelan tidak mempertimbangkan kategori produk, toko, maupun variabel eksternal lainnya. 

- b. Data yang digunakan merupakan data penjualan historis dari sumber publik atau dataset terbuka yang tersedia secara daring. 

- c. Model _baseline_ yang digunakan dalam ensemble terdiri dari tiga model, yaitu _Moving Average (MA)_ , _Exponential Smoothing (ES)_ , dan _Recurrent Neural Network (RNN)_ 

- d. Algoritma optimasi yang digunakan adalah _Grey Wolf Optimizer (GWO)_ dengan MAPE sebagai fungsi objektif. 

- e. Evaluasi kinerja model dilakukan menggunakan metrik _Mean Absolute Percentage Error_ (MAPE), _Mean Absolute Error_ (MAE), _Mean Squared Error_ (MSE), _Root Mean Squared Error_ (RMSE), dan _R-squared_ (R²) 

##### **1.2.3 Rumusan Masalah** 

Berdasarkan identifikasi masalah di atas maka dapat dirumuskan masalah 

yaitu: 

- a. Bagaimana performa _Moving Average_ , _Exponential Smoothing_ , dan _Recurrent Neural Network_ dan model _ensemble_ yang dioptimasi dengan GWO dalam peramalan penjualan? 

- b. Model manakah di antara metode _Moving Average_ , _Exponential Smoothing_ , _Recurrent Neural Network,_ dan model _ensemble_ yang dioptimasi dengan GWO yang memberikan akurasi terbaik pada dataset Store Sales - Time Series Forecasting dari Kaggle? 

UNIVERSITAS PAMULANG 

4 

- c. Berapa peningkatan akurasi hasil peramalan yang dihasilkan oleh model _ensemble_ yang dioptimasi GWO dibandingkan dengan model _baseline_ terbaik? 

- d. Apakah algoritma GWO cukup stabil dan konsisten dalam optimasi bobot _ensemble_ ? 

##### **1.3 Tujuan dan Manfaat Penelitian** 

Adapun tujuan penulis melakukan penelitian ini adalah sebagai berikut: 

- a. Mengembangkan dan membandingkan model _ensemble learning_ untuk peramalan penjualan yang menggabungkan model _Moving Average_ , _Exponential Smoothing_ , dan _Recurrent Neural Network_ 

- b. Menerapkan algoritma _Grey Wolf Optimizer (GWO)_ untuk menentukan bobot optimal model _ensemble_ guna memperoleh hasil prediksi dengan kesalahan minimum 

- c. Mengidentifikasi dan membandingkan kinerja model _ensemble_ dan membandingkannya dengan model _baseline_ berdasarkan metrik _Mean Absolute Percentage Error_ (MAPE), _Mean Absolute Error_ (MAE), _Mean Squared Error_ (MSE), _Root Mean Squared Error_ (RMSE), dan _R-squared_ (R²) 

- d. Menganalisa stabilitas dan konsistensi algoritma GWO dalam optimasi bobot ensemble melalui pengujian _multi-run_ 

Sedangkan manfaat dari penelitian ini adalah: 

- a. Manfaat Akademis: 

   - Memberikan kontribusi ilmiah dalam pengembangan metode hybrid yang menggabungkan _ensemble learning_ dan algoritma metaheuristik _Grey Wolf Optimizer (GWO)_ untuk peramalan penjualan, serta membuka peluang pengembangan lebih lanjut dengan arsitektur _deep learning_ yang lebih mutakhir seperti _Long Short-Term Memory (LSTM)_ , _Gated Recurrent Unit (GRU)_ , atau _Transformer_ guna 

UNIVERSITAS PAMULANG 

5 

menangkap pola ketergantungan jangka panjang yang lebih kompleks pada data penjualan. 

- b. Manfaat Praktis: 

Menyediakan alternatif solusi awal berbasis pendekatan _univariate_ yang dapat dikembangkan lebih lanjut menjadi sistem peramalan _multivariate_ dengan mengintegrasikan variabel-variabel bisnis seperti harga produk, promosi, hari libur, kategori produk, dan wilayah toko. Hal ini memungkinkan perusahaan memperoleh prediksi penjualan yang lebih granular dan akurat, sehingga mendukung pengambilan keputusan strategis seperti perencanaan stok, alokasi promosi, dan strategi harga yang lebih tepat sasaran. 

##### **1.4 Sistematika Penulisan** 

Sistematika penulisan menjelaskan tentang berapa penjelasan singkat isi dari masing-masing bab dalam proposal tesis ini sebagai berikut: 

##### **BAB I  PENDAHULUAN** 

Bab ini mengemukakan latar belakang masalah yang diteliti, identifikasi masalah, rumusan masalah, tujuan penelitian, batasan masalah, manfaat penelitian, metode penelitian dan sistematika penulisan. 

##### **BAB II  LANDASAN TEORI DAN KERANGKA PEMIKIRAN** 

Bab ini mencakup beberapa sub bab antara lain : tinjauan pustaka, teori teori yang mendukung topik dan kerangka pemikiran. 

##### **BAB III  METODE PENELITIAN** 

Bab ini mencakup beberapa sub bab antara lain: analisis kebutuhan, perancangan penelitian serta Teknik analisis 

UNIVERSITAS PAMULANG 

6 

#### **BAB II** 

#### **LANDASAN TEORI DAN KERANGKA PEMIKIRAN** 

##### **2.1 Tinjauan Pustaka** 

Penelitian mengenai optimasi model _ensemble_ /gabungan untuk peramalan/prediksi bukanlah baru pertama kali dilakukan. Berikut penelitian terdahulu yang relevan dengan penelitian ini sebagaimana tertera di Tabel 2.1. 

Tabel 2.1 Literatur tentang penelitian sebelumnya 

|No.|1|
|---|---|
|Peneliti|Xinti Sun, Minyu Nong, Fei Meng, Xiaojuan Sun,<br>Lihe Jiang, Zihao Li, dan Peng Zhang|
|Tahun|2024|
|Judul|_Architecting the metabolic reprogramming survival_<br>_risk framework in LUAD through single-cell_<br>_landscape analysis: three-stage ensemble learning_<br>_with genetic algorithm optimization_|
|Metode Optimasi|_Genetic Algorithm_(GA)|
|Model ML/DL|_Cox Regression, Random Survival Forest, CoxBoost,_<br>_Gradient Boosting Machine, Support Vector Machine_|
|Model_Ensemble_|_Stacking_|
|Hasil|Model_Ensemble_dengan optimasi_Genetic Algorithm_<br>memberikan akurasi tertinggi, jauh melampaui<br>performa model individual|
|No.|2|
|Peneliti|Ummey Hany Ainan, Lip Yee Por, Yen-Lin Chen,<br>Jing Yang, dan Chin Soon Ku|
|Tahun|2024|
|Judul|_Advancing Bankruptcy Forecasting With Hybrid_<br>_Machine Learning Techniques: Insights From an_<br>_Unbalanced Polish Dataset_|



UNIVERSITAS PAMULANG 

7 

|Metode Optimasi|_Genetic Algorithm_(GA) dan_Particle Swarm_<br>_Optimization_(PSO)|
|---|---|
|Model ML/DL|_Random Forest_(RF),_Support Vector Machine_<br>(SVM),_XGBoost_|
|Model_Ensemble_|_Stacking_|
|Hasil|Kombinasi_Ensemble_ _XGBoost (structured learning)_<br>dan_ANN (nonlinear learning)_dengan GA mempunyai<br>akurasi paling baik.|
|No.|3|
|Peneliti|Abdulrahman A. Alghamdi, Abdelhameed Ibrahim,<br>El-Sayed M. El-Kenawy, Abdelaziz A. Abdelhamid|
|Tahun|2023|
|Judul|_Renewable Energy Forecasting Based on Stacking_<br>_Ensemble Model and Al-Biruni Earth Radius_<br>_Optimization Algorithm_|
|Metode Optimasi|GABER (_Genetic Algorithm + Al-Biruni Earth Radius_<br>_Optimization_) — gabungan_Genetic Algorithm (GA)_<br>untuk memperkuat eksplorasi dan mutasi dengan_Al-_<br>_Biruni Earth Radius (BER)_untuk eksploitasi|
|Model ML/DL|LSTM, Bidirectional LSTM (BiLSTM), Hermite<br>Neural Network (HNN)|
|Model_Ensemble_|_Stacking_|
|Hasil|Model GABER-HNN_stacking ensemble_memberikan<br>akurasi sangat tinggi untuk prediksi energi terbarukan<br>mengungguli semua model individu baik pada dataset<br>angin maupun radiasi surya.|
|No.|4|
|Peneliti|Eslam A. Aly, El-Sayed M. El-Kenawy, Ahmed E.<br>Hassanien, Sami Elhoseny, Mohamed M. Ghoneim,<br>dan Abdelaziz A. Abdelhamid|
|Tahun|2023|



UNIVERSITAS PAMULANG 

8 

|Judul|_Enhancing photovoltaic power prediction using an_<br>_adaptive hybrid model based on improved gorilla_<br>_troops optimizer and neural networks_|
|---|---|
|Metode Optimasi|_Improved Gorilla Troops Optimizer_(IGTO)|
|Model ML/DL|ANN|
|Model_Ensemble_|_-_|
|Hasil|IGTO-ANN mengungguli PSO-ANN, GA-ANN,<br>GWO-ANN, MPA-ANN, SSA-ANN, dan ANN klasik<br>tanpa optimasi|
|No.|5|
|Peneliti|Zheng Wang, Yihui Zhang, Yuanyuan Ma, Jin Wu,<br>dan Zhongxiang Liu|
|Tahun|2022|
|Judul|_A deep learning and ensemble learning based_<br>_architecture for metro passenger flow prediction_|
|Metode Optimasi|_Bayesian Optimization_(BO) digunakan untuk tuning<br>hyperparameter pada model ensemble hybrid (mis.<br>jumlah neuron, learning rate, dropout)|
|Model ML/DL|CNN, LSTM, GRU, dan_Gradient Boosting Decision_<br>_Tree_(GBDT)|
|Model_Ensemble_|_Stacking_|
|Hasil|Model_Ensemble_memiliki akurasi paling baik untuk<br>prediksi arus penumpang metro dibanding model DL<br>tunggal.|
|No.|6|
|Peneliti|Mahya Seyedan, Pereshteh Mafakheri, Chun Wang|
|Tahun|2023|
|Judul|_Order-up-to-level inventory optimization model using_<br>_time-series demand forecasting with ensemble deep_<br>_learning_|
|Metode Optimasi|-|



UNIVERSITAS PAMULANG 

9 

|Model ML/DL|MLP, LSTM, 1D CNN|
|---|---|
|Model_Ensemble_|_Stacking_|
|Hasil|_Ensemble deep learning_dengan model heterogen<br>meningkatkan akurasi peramalan dan optimasi<br>inventaris dibandingkan model individual|
|No.|7|
|Peneliti|Anne Carolina Rodrigues Klaar, Stefano Frizzo<br>Stefanon, Laio Oriel Seman, Viviana Cocco Mariani,<br>Leandro dos Santos Coelho|
|Tahun|2023|
|Judul|_Structure_<br>_Optimization_<br>_of_<br>_Ensemble_<br>_Learning_<br>_Methods and Seasonal Decomposition Approaches to_<br>_Energy Price Forecasting in Latin America: A Case_<br>_Study about Mexico_|
|Metode Optimasi|Optuna dengan Tree-structured Parzen Estimator<br>(TPE), untuk menemukan struktur ensemble terbaik<br>(kombinasi model, bobot, dan jumlah lag)|
|Model ML/DL|_Adaptive_<br>_Boosting_<br>_(AdaBoost),_<br>_Bootstrap_<br>_Aggregation_<br>_(Bagging),_<br>_Gradient_<br>_Boosting,_<br>_Histogram-Based Gradient Boosting, Random Forest_|
|Model_Ensemble_|_Voting_|
|Hasil|Optimasi struktur ensemble dengan Optuna + SDMA<br>sangat efektif untuk forecasting harga energi, selain itu<br>pendekatan_ensemble_menghasilkan akurasi yang jauh<br>lebih tinggi dari pada model individual|
|No.|8|
|Peneliti|Ahmed Ali Mohamed Warad, Khaled Wassif, Nagy<br>Ramadan Darwish|
|Tahun|2024|
|Judul|_An ensemble learning model for forecasting water-_<br>_pipe leakage_|



UNIVERSITAS PAMULANG 

10 

|Metode Optimasi|Bayesian<br>Optimization<br>untuk<br>optimasi<br>hyperparameter|
|---|---|
|Model ML/DL|Regression Trees (RT)|
|Model_Ensemble_|_-_|
|Hasil|Optimasi<br>hyperparameter<br>dengan<br>Bayesian<br>optimization secara signifikan meningkatkan akurasi<br>prediksi|
|No.|9|
|Peneliti|Seyed Matin Malakouti, Farrokh Karimi, Hamid<br>Abdollahi,<br>Mohammad<br>Bagher<br>Menhaj,<br>Amir<br>Abolfazl Suratgar, Mohammad Hassan Moradi|
|Tahun|2024|
|Judul|_Advanced techniques for wind energy production_<br>_forecasting: Leveraging multi-layer Perceptron +_<br>_Bayesian optimization, ensemble learning, and CNN-_<br>_LSTM models_|
|Metode Optimasi|_Bayesian Optimization_untuk tuning hyperparameter<br>MLP dan Grid Search CV untuk XGBoost dan<br>Ensemble|
|Model ML/DL|_XGBoost, Multi-Layer Perceptron + Bayesian_<br>_Optimization_(MLP + BO),_Gradient Boosting_|
||_Regression_<br>_Tree_<br>(GBDT),<br>_Ensemble_|
||_Learning_(gabungan<br>Gradient<br>Boosting<br>dan|
||XGBoost),<br>CNN-LSTM (_Convolutional_<br>_Neural_<br>_Network + Long Short-Term Memory_)|
|Model_Ensemble_|_Voting_|
|Hasil|Model_hybrid_dan_ensemble_menunjukkan performa<br>lebih baik dibanding model tunggal|
|No.|10|
|Peneliti|Karlo Abnoosian, Rahman Farnoosh, Mohammad<br>Hassan Behzadi|



UNIVERSITAS PAMULANG 

11 

|Tahun|2023|
|---|---|
|Judul|_Prediction of diabetes disease using an ensemble of_<br>_machine learning multi-classifier models_|
|Metode Optimasi|_Bayesian Optimization_untuk SVM, DT, RF dan<br>_Grid Search_untuk k-NN, GNB, AdaBoost|
|Model ML/DL|k-Nearest Neighbors (KNN), Support Vector Machine<br>(SVM), Decision Tree (DT), Random Forest (RF),<br>AdaBoost (AB), Gaussian Naive Bayes (GNB)|
|Model_Ensemble_|_Weighted Voting_dengan bobot berdasarkan nilai AUC<br>setiap model dasar|
|Hasil|Model_ensemble_mengungguli semua model tunggal<br>dan model hybrid sebelumnya yang diusulkan di<br>dataset yang sama|
|No.|11|
|Peneliti|Weifang Liang, Yong Liu, Simon Somogyi, David<br>Anderson|
|Tahun|2024|
|Judul|_A Multi-Model, Ensemble Approach to Forecasting_<br>_United States Food Prices_|
|Metode Optimasi|_Maximum a Posteriori_(MAP) dengan algoritma L-<br>BFGS untuk optimasi hyperparameter|
|Model ML/DL|_ARIMA, Exponential Smoothing, Local Linear_<br>_Regression, Gaussian Process_(GP)|
|Model_Ensemble_|_Weighted Average_|
|Hasil|Model<br>_ensemble_mengungguli<br>ARIMA<br>dalam<br>hal MAPE dan RMSE total<br>dengan<br>memberikan<br>prediksi harga pangan yang lebih akurat|
|No.|12|
|Peneliti|Bohdan M. Pavlyshenko|
|Tahun|2019|



UNIVERSITAS PAMULANG 

12 

|Judul|_Machine-Learning Models for Sales Time Series_<br>_Forecasting_|
|---|---|
|Metode Optimasi|-|
|Model ML/DL|_Random Forest, ExtraTree, Lasso Regression, Neural_<br>_Network, ARIMA, XGBoost_|
|Model_Ensemble_|_Stacking_|
|Hasil|_Stacking_ _ensemble_meningkatkan akurasi prediksi<br>14% lebih tinggi dari model individu|
|No.|13|
|Peneliti|Tian Jin|
|Tahun|2025|
|Judul|_Optimizing Retail Sales Forecasting Through a PSO-_<br>_Enhanced Ensemble Model Integrating LightGBM,_<br>_XGBoost, and Deep Neural Networks_|
|Metode Optimasi|_Particle Swarm Optimization_(PSO) digunakan untuk<br>tuning hyperparameter dan optimasi bobot_ensemble_|
|Model ML/DL|LightGBM, XGBoost, Deep Neural Networks (DNN)|
|Model_Ensemble_|_Weighted Average_|
|Hasil|Model_ensemble_dengan PSO mencapai performa<br>terbaik,<br>dengan<br>bobot<br>terbanyak<br>oleh<br>DNN<br>dibandingkan dengan model tunggal atau_ensemble_<br>tanpa PSO.|
|No.|14|
|Peneliti|Islam M. Hammam, Amin K. El-Kharbotly, Yonma<br>M. Sadek|
|Tahun|2025|
|Judul|_Adaptive_<br>_demand_<br>_forecasting_<br>_framework_<br>_with_<br>_weighted ensemble of regression and machine_<br>_learning models along life cycle variability_|



UNIVERSITAS PAMULANG 

13 

|Metode Optimasi|_Grid Search_untuk tuning hyperparameter XGBoost<br>dan untuk menentukan bobot_ensemble_dengan<br>meminimalkan RMSE|
|---|---|
|Model ML/DL|AR, ARMA, ARIMA, SARIMA, XGBoost|
|Model_Ensemble_|_Weighted Average_|
|Hasil|Model_ensemble_memberikan performa terbaik pada<br>dataset dengan pola campuran dengan bobot SARIMA<br>(0,3) dan XGBoost (0,7), 80% lebih akurat daripada<br>model ARIMA tunggal.|
|No.|15|
|Peneliti|Dr. K. Alice, Syed Hamad ul Haq Andrabi, Shambhavi<br>Jha|
|Tahun|Tidak disebutkan secara eksplisit|
|Judul|_Sales forecasting based on Ensemble Predictions_|
|Metode Optimasi|Tidak disebutkan secara spesifik|
|Model ML/DL|XGBoost, LightGBM (LGBM), CatBoost|
|Model_Ensemble_|_Voting Regressor_dengan pembobotan berdasarkan<br>nilai MAE masing-masing model|
|Hasil|_Ensemble Voting Regressor_memiliki akurasi tertinggi|
||dengan<br>akurasi<br>98,7%<br>meningkat<br>0,001%<br>dibandingkan model individu|
|No.|16|
|Peneliti|Mustapha Ismail, Hafsat Muhammad Tukur, Mamudu<br>Friday|
|Tahun|2025|
|Judul|_Sales Prediction using Ensemble Machine Learning_<br>_Model_|
|Metode Optimasi|_Grid Search_untuk hyperparameter tuning|
|Model ML/DL|_Random Forest_(RF),_XGBoost, Support Vector_<br>_Machine_(SVM)|
|Model_Ensemble_|_Stacking_|



UNIVERSITAS PAMULANG 

14 

|Hasil|_Stacking Ensemble_mencapai performa terbaik dengan<br>R² = 0.9990, lebih tinggi 0,0006 dibandingkan model<br>individu.|
|---|---|
|No.|17|
|Peneliti|Nagaraju Jajam, Nagendra Panini Challa, Kamepalli<br>S. L. Prasanna, C H Venkata Sasi Deepthi|
|Tahun|2023|
|Judul|_Arithmetic_<br>_Optimization_<br>_With_<br>_Ensemble_<br>_Deep_<br>_Learning SBLSTM-RNN-IGSA Model for Customer_<br>_Churn Prediction_|
|Metode Optimasi|_Arithmetic Optimization Algorithm_(AOA) untuk<br>ekstraksi fitur dan_Improved Gravitational Search_|
||_Optimization_<br>_Algorithm_<br>(IGSA) untuk<br>tuning<br>hyperparameter|
|Model ML/DL|SBLSTM-RNN (_Stacked Bidirectional Long Short-_<br>_Term Memory – Recurrent Neural Network_)|
|Model_Ensemble_|_Stacking_|
|Hasil|Model AOA-SBLSTM-RNN-IGSA mencapai akurasi<br>tertinggi mengungguli semua model pembanding<br>(baik klasik maupun deep learning) dan cocok untuk<br>aplikasi prediksi churn di sektor asuransi|
|No.|18|
|Peneliti|Zongxi Qu, Yutong Li, Xia Jiang, Chunhua Niu|
|Tahun|2023|
|Judul|_An innovative ensemble model based on multiple_<br>_neural networks and a novel heuristic optimization_<br>_algorithm for COVID-19 forecasting_|
|Metode Optimasi|SCWOA (_Sine Cosine Algorithm-Whale Optimization_<br>_Algorithm_)|
|Model ML/DL|_Back-Propagation Neural Network_(BPNN),_Elman_|
||_Neural Network_(ENN),_Adaptive Neuro-Fuzzy_|



UNIVERSITAS PAMULANG 

15 

||_Inference System_(ANFIS),_Long Short-Term Memory_<br>(LSTM)|
|---|---|
|Model_Ensemble_|_Weighted Ensemble_|
|Hasil|SCWOA unggul dibandingkan PSO, GWO, SCA, dan<br>WOA dalam pengujian 15 fungsi benchmark, model<br>_ensemble_berbasis SCWOA yang mengkombinasikan<br>BPNN, ENN, ANFIS, dan LSTM terbukti sangat<br>efektif<br>dan<br>robust untuk<br>prediksi<br>COVID-19,<br>mengatasi<br>keterbatasan<br>model<br>individu<br>dan<br>meningkatkan akurasi prediksi secara signifikan|
|No.|19|
|Peneliti|S. Vanitha, P. Balasubramanie|
|Tahun|2023|
|Judul|_Improved Ant Colony Optimization and Machine_<br>_Learning Based Ensemble Intrusion Detection Model_|
|Metode Optimasi|Improved Ant Colony Optimization (IACO)|
|Model ML/DL|_Distance Decision Tree_(DDT),_Adaptive Neuro-Fuzzy_<br>_Inference System_(ANFIS),_Mahalanobis Distance_<br>_Support Vector Machine_(MDSVM)|
|Model_Ensemble_|_Weighted Majority Voting_|
|Hasil|IACO berhasil meningkatkan seleksi fitur dan kinerja<br>deteksi intrusi, model_ensemble_mengungguli model<br>individu (DT, SVM,_Ensemble_sederhana) dalam hal<br>akurasi.|
|No.|20|
|Peneliti|Geetha Narasimhan, Akila Victor|
|Tahun|2024|
|Judul|_Bio-inspired disease prediction: harnessing the power_<br>_of electric eel foraging optimization algorithm with_<br>_machine learning for heart disease prediction_|



UNIVERSITAS PAMULANG 

16 

|Metode Optimasi|_Electric Eel Foraging Optimization Algorithm_<br>(EEFOA)|
|---|---|
|Model ML/DL|_Random Forest_(RF) dan_K-Nearest Neighbors_(KNN)|
|Model_Ensemble_|_Bagging_|
|Hasil|EEFOA berhasil meningkatkan kualitas feature<br>selection dibanding metaheuristic lain. Kombinasi<br>EEFOA dan_Random Forest_memberikan performa<br>terbaik.|
|No.|21|
|Peneliti|Mohammed<br>Husayn,<br>Oluwatayomi<br>Rereloluwa<br>Adegboye, Ahmad Alzubi|
|Tahun|2025|
|Judul|_GWO-Optimized Ensemble Learning for Interpretable_<br>_and Accurate Prediction of Student Academic_<br>_Performance in Smart Learning Environments_|
|Metode Optimasi|_Grey Wolf Optimizer_(GWO)|
|Model ML/DL|CatBoost (GBDT), Extra Tree, Random Forest|
|Model_Ensemble_|_Weighted average_|
|Hasil|Ensemble heterogen yang dioptimasi dengan GWO<br>mengungguli semua model individual.|
|No.|22|
|Peneliti|Geetha Narasimhan, Akila Victor|
|Tahun|2024|
|Judul|_Grey wolf optimized stacked ensemble machine_<br>_learning based model for enhanced efficiency and_<br>_reliability of predicting early heart disease_|
|Metode Optimasi|_Grey Wolf Optimizer_(GWO)|
|Model ML/DL|_Logistic Regression_(LR), KNN, SVM / SVC,_Naive_<br>_Bayes_(NB),_Random Forest_(RF),_Decision Tree_<br>(DT/CART), MLP (Neural Network),_Extra Trees_,<br>XGBoost, SGD,_AdaBoost_,_Gradient Boosting_(GBM)|



UNIVERSITAS PAMULANG 

17 

|Model_Ensemble_|_Stacking_|
|---|---|
|Hasil|Kombinasi<br>GWO<br>dan<br>_stacking_<br>_ensemble_<br>meningkatkan akurasi signifikan yang menghasilkan<br>model yang lebih stabil yaitu MCC tinggi, Log_Loss<br>rendah sehingga cocok untuk early heart disease<br>prediction.|
|No.|23|
|Peneliti|R. Vinothkumar, S. Kannan, M. Elhoseny|
|Tahun|2023|
|Judul|_Stacked ensemble learning based intrusion detection_<br>_model_<br>_for_<br>_IoT_<br>_using_<br>_feature_<br>_selection_<br>_and_<br>_hyperparameter optimization_|
|Metode Optimasi|_Grey Wolf Optimizer_(GWO)|
|Model ML/DL|_Decision Tree_(DT),_Random Forest_(RF),_XGBoost_,<br>KNN, SVM,_Meta-learner_(_Logistic Regression_)|
|Model_Ensemble_|_Stacking_|
|Hasil|Model_stacking ensemble_dan GWO menghasilkan<br>akurasi tertinggi dibanding model individual dengan<br>akurasi  lebih 99% pada beberapa skenario dataset|
|No.|24|
|Peneliti|Eslam Hamouda, Mayada Tarek|
|Tahun|2024|
|Judul|_A hybrid approach of ensemble learning and grey wolf_<br>_optimizer for DNA splice junction prediction_|
|Metode Optimasi|_Grey Wolf Optimizer_(GWO)|
|Model ML/DL|SVM, Decision Tree, KNN, Naïve Bayes|
|Model_Ensemble_|_Two-layer hybrid ensemble_|
|Hasil|Akurasi terbaik: 96.63% (5-fold CV, GWO+SVM)<br>Mengungguli metode sebelumnya (CNN 90.25%,<br>CNN+BLSTM 96.00%)|
|No.|25|



UNIVERSITAS PAMULANG 

18 

|Peneliti|SVSV Prasad Sanaboina, Dr. M. Chandra Naik, Dr. K.<br>Rajiv|
|---|---|
|Tahun|2025|
|Judul|_An Advanced Ensemble Framework Employing Grey_<br>_Wolf Optimization and Feature Selection Techniques_<br>_for Enhanced Intrusion Detection on Unbalanced_<br>_NSL-KDD Data_|
|Metode Optimasi|_Grey Wolf Optimizer_(GWO)|
|Model ML/DL|_Decision Tree_(DT),_Support Vector Machine_(SVM)<br>_Naïve Bayes_,_K-Nearest Neighbors_(KNN),_Logistic_<br>_Regression_|
|Model_Ensemble_|_Majority Voting, Stacking (DT + KNN_sebagai model<br>terbaik_),_Bobot ensemble dioptimasi menggunakan<br>_GWO_|
|Hasil|_Stacking ensemble_(DT-KNN + GWO) menghasilkan<br>performa terbaik mengungguli model individual|
|No.|26|
|Peneliti|Zheng Zhang, Yongjie Li, Jianhua Gu|
|Tahun|2023|
|Judul|_Intrusion detection model based on improved grey_<br>_wolf optimization and ensemble learning_|
|Metode Optimasi|_Improved Grey Wolf Optimizer_(IGWO)|
|Model ML/DL|_Support Vector Machine_(SVM),_K-Nearest Neighbor_<br>(KNN),_Decision Tree_(DT),_Random Forest_(RF)|
|Model_Ensemble_|_Stacking_|
|Hasil|Model_stacking ensemble_mempunyai akurasi > 99% ,<br>lebih unggul dibanding model individual, serta<br>penggunaan IGWO meningkatkan konvergensi dan<br>stabilitas dibandingkan GWO standar|
|No.|27|



UNIVERSITAS PAMULANG 

19 

|Peneliti|Muhammad Afifudin, Ahmad Junaedi, Andreas<br>Nugroho, Izzatul Fithriyah|
|---|---|
|Tahun|2024|
|Judul|_GWO-SVM: An Approach to Improving SVM_<br>_Performance_<br>_Using_<br>_Grey_<br>_Wolf_<br>_Optimizer_<br>_in_<br>_Intellectual Disability Classification_|
|Metode Optimasi|_Grey Wolf Optimizer_(GWO) untuk Optimasi<br>parameter SVM (C, Learning Rate, Max Iteration)|
|Model ML/DL|Support Vector Machine (SVM)|
|Model_Ensemble_|_-_|
|Hasil|Akurasi GWO-SVM lebih tinggi 4.8% dibandingkan<br>SVM 90.47% dengan parameter terbaik: C=2.40,<br>_Learning Rate_=0.0560,_Max Iteration_=144|
|No.|28|
|Peneliti|Lihua Lou, Weidong Xia, Zhen Sun, Shichao Quan,<br>Shaobo Yin, Zhihong Gao, Cai Lin|
|Tahun|2023|
|Judul|_COVID-19 mortality prediction using ensemble_<br>_learning and grey wolf optimization_|
|Metode Optimasi|_Grey Wolf Optimizer_(GWO) untuk optimasi bobot<br>_weighted ensemble_|
|Model ML/DL|_Gradient Boosting_(GB),_Random Forest_(RF),<br>_Extremely Randomized Trees_(ERT)|
|Model_Ensemble_|_Weighted ensemble_|
|Hasil|Model ensemble dengan 10 fitur terpilih mempunyai<br>AUC terbaik: 0.7802 dan GWO efektif untuk optimasi<br>bobot_ensemble_.|
|No.|29|
|Peneliti|Manokaran, J.Vairavel, G.|
|Tahun|2023|



UNIVERSITAS PAMULANG 

20 

|Judul|_IGWO-SoE: Improved Grey Wolf Optimization-Based_<br>_Stack of Ensemble Learning Algorithm for Anomaly_<br>_Detection in Internet of Things Edge Computing_|
|---|---|
|Metode Optimasi|_Improved Grey Wolf Optimizer_(IGWO)|
|Model ML/DL|_Random Forest_(RF),_Support Vector Machine_<br>(SVM), k-Nearest Neighbor (KNN),_Decision Tree_<br>(DT),_Logistic Regression_|
|Model_Ensemble_|_Stacking_|
|Hasil|Akurasi model ensemble mencapai lebih dari 99%<br>pada NSL-KDD, dengan peningkatan detection rate<br>dan penurunan false alarm rate dibanding model<br>individual. IGWO meningkatkan konvergensi dan<br>stabilitas dibanding GWO standar.|
|No.|30|
|Peneliti|ZhiQiang Zeng, Saratha Sathasivam, Jing Xin, Huan<br>Zhao|
|Tahun|2026|
|Judul|_Real-time dynamic prediction of HFMD transmission_<br>_using SEIRQ-ARIMA hybrid model optimized by_<br>_multi-stage ABC-GWO algorithm_|
|Metode Optimasi|_Multi-stage Artificial Bee Colony - Grey Wolf_<br>_Optimizer_(ABC-GWO)|
|Model ML/DL|SEIRQ, ARIMA|
|Model_Ensemble_|_Weighted_|
|Hasil|Model_ensemble_SEIRQ-ARIMA yang dioptimasi<br>ABC-GWO mengungguli model SEIRQ, ARIMA,<br>dan SEIRQ+LSTM dengan RMSE 14,2|



Beberapa literatur tersebut di atas menunjukkan bahwa _Grey Wolf Optimizer_ (GWO) semakin luas digunakan dalam dua konteks utama pada machine learning, yaitu _hyperparameter tuning_ dan optimasi bobot _ensemble_ . 

UNIVERSITAS PAMULANG 

21 

Dalam konteks _hyperparameter tuning_ , GWO dimanfaatkan untuk mencari kombinasi parameter optimal yang memaksimalkan kinerja model tanpa memerlukan informasi gradien, sementara itu dalam konteks optimasi bobot _ensemble_ GWO dimanfaatkan untuk memberikan bobot kontribusi masingmasing model individual sehingga mencapai fungsi obyektif yang diharapkan. Studi oleh Y.K. Saheed mengintegrasikan GWO pada model stacking untuk _intrusion detection_ berbasis IoT, di mana GWO digunakan untuk _feature selection_ sekaligus _hyperparameter tuning_ sehingga menghasilkan akurasi di atas 99% (Saheed & Misra, 2024). Pendekatan serupa dilakukan oleh Manokaran (2023) melalui Improved GWO (IGWO) yang meningkatkan konvergensi dan stabilitas parameter model IDS (Manokaran & Vairavel, 2023). Pada klasifikasi disabilitas intelektual, Muhammad Afifudin (2024) menunjukkan bahwa GWO efektif dalam mengoptimasi parameter SVM (C, _learning rate_ , dan maksimum iterasi), meningkatkan akurasi secara signifikan dibanding SVM standar (Afifudin et al., 2024). Sementara itu, ZhiQiang Zeng (2026) mengembangkan pendekatan hybrid multi-stage ABC-GWO untuk mengoptimasi parameter model SEIRQ-ARIMA dalam prediksi penyebaran HFMD secara real-time, menghasilkan reduksi error yang substansial. Temuan-temuan ini menegaskan bahwa mekanisme eksplorasi–eksploitasi GWO efektif dalam menjelajahi ruang parameter yang kompleks dan non-linear (ZhiQiang Zeng, Saratha Sathasivam & We, 2026). 

Di sisi lain, GWO juga banyak digunakan untuk optimasi bobot _ensemble_ , terutama dalam _weighted average_ maupun _stacking ensemble_ . Penelitian oleh Mohammed Husayn (2025) memanfaatkan GWO untuk mengoptimasi bobot _weighted ensemble_ berbasis _CatBoost, Extra Tree_ , dan _Random Forest_ dalam prediksi performa akademik, menghasilkan model yang lebih akurat dibandingkan model individual (Husayn et al., 2025). Studi SVSV Prasad Sanaboina (2025) menerapkan GWO untuk mengoptimasi bobot pada framework _ensemble intrusion detection_ berbasis data tidak seimbang, meningkatkan stabilitas dan akurasi sistem (SVSV Prasad Sanaboina, M Chandra Naik, K Rajiv, 2025). Dalam domain kesehatan, Lihua Lou (2023) menggunakan GWO untuk menentukan bobot optimal pada _weighted ensemble_ 

UNIVERSITAS PAMULANG 

22 

( _Gradient Boosting, Random Forest_ , ERT) dalam prediksi mortalitas COVID19, menunjukkan performa robust dengan AUC optimal dan variansi rendah(Lou et al., 2023). Pendekatan stacking berbasis GWO juga diterapkan oleh Geetha Narasimhan (2024) untuk prediksi penyakit jantung dini (Narasimhan & Victor, 2024), serta oleh Eslam Hamouda (2024) pada prediksi DNA splice junction, di mana GWO meningkatkan stabilitas dan akurasi dibanding metode sebelumnya(Hamouda & Tarek, 2024). Selain itu, Manokaran (2023) mengembangkan IGWO-SoE untuk _anomaly detection_ pada IoT _edge computing_ , membuktikan bahwa optimasi kombinasi model berbasis GWO mampu meningkatkan _detection rate_ dan menurunkan _false alarm rate_ (Manokaran & Vairavel, 2023). Secara keseluruhan, literatur tersebut menunjukkan bahwa GWO tidak hanya efektif untuk _hyperparameter tuning_ model tunggal, tetapi juga sangat potensial dalam menyelesaikan permasalahan optimasi bobot _ensemble_ . Hal ini mengindikasikan bahwa GWO memiliki kontribusi dalam skenario optimasi pada _machine learning_ , sekaligus membuka peluang penelitian lebih lanjut, khususnya pada domain peramalan penjualan dan optimasi _weighted ensemble_ berbasis time series. 

Berdasarkan berbagai studi tersebut membuktikan efektivitas GWO dalam optimasi bobot _ensemble_ , terdapat pola umum yang dapat diidentifikasi. Pertama, hampir seluruh penelitian berada dalam domain klasifikasi, bukan regresi. Kedua, dataset yang digunakan umumnya berupa data statis ( _crosssectional_ ), bukan data _time series_ dengan komponen tren dan musiman. Ketiga, sedikit penelitian yang mengkaji secara khusus optimasi bobot _ensemble_ pada metode peramalan berbasis _Moving Average_ , _Exponential Smoothing_ , atau _Recurrent Neural Network_ . 

Literatur yang ada belum secara eksplisit menginvestigasi penggunaan GWO untuk mengoptimasi bobot kombinasi model baseline dalam konteks peramalan penjualan ritel atau manufaktur. Sehingga terdapat celah penelitian pada penerapan GWO untuk optimasi bobot _weighted ensemble_ dalam peramalan penjualan. Celah ini menjadi landasan bagi penelitian yang diusulkan, dengan hipotesis bahwa mekanisme eksplorasi–eksploitasi GWO yang seimbang mampu menghasilkan kombinasi bobot yang lebih optimal 

UNIVERSITAS PAMULANG 

23 

dibandingkan _equal-weight ensemble_ maupun metode heuristik sederhana, sehingga meningkatkan akurasi dan stabilitas peramalan penjualan dalam konteks bisnis nyata. 

##### **2.2 Landasan Teori** 

Dalam landasan teori, ada beberapa pembahasan yang akan disajikan berhubungan dengan penelitian yang dilakukan, teori-teori yang akan dikemukakan dalam penelitian ini akan  menjelaskan tentang apa saja yang akan diteliti dan menjelaskan secara teori mengenai metode yang digunakan. 

##### **2.2.1 Peramalan (** **_Forecasting_ )** 

Peramalan ( _forecasting_ ) merupakan suatu upaya sistematis yang dilakukan untuk memperkirakan kebutuhan di masa mendatang, baik dalam hal kuantitas, kualitas, waktu, maupun lokasi yang dibutuhkan untuk pemenuhan barang atau jasa. Peramalan sebagai seni dan ilmu untuk memperkirakan kejadian di masa depan dengan melibatkan pengambilan data historis dan memproyeksikannya ke masa mendatang melalui suatu bentuk model matematis (Heizer et al., 2014). Hasil dari proses peramalan ini kemudian dijadikan sebagai acuan fundamental bagi para pengambil keputusan dalam merumuskan berbagai kebijakan strategis maupun operasional(Heizer et al., 2014) . 

Dalam dinamika dunia usaha, ketidakstabilan kondisi ekonomi, pasar, dan perilaku konsumen seringkali menjadi tantangan utama dalam menyusun perencanaan yang efektif. Di sinilah peran krusial peramalan, yaitu untuk membantu para pengambil keputusan dalam mengurangi ketidakpastian. Dengan adanya peramalan yang akurat, perusahaan dapat merencanakan kapasitas produksi, menyusun anggaran, merencanakan penjualan, mengelola persediaan (inventory), serta merencanakan pengadaan bahan baku secara lebih efisien (Rob J. Hyndman & Athanasopoulos, 2018) . Peramalan menjadi inti dari efisiensi aktivitas manufaktur dan jasa, karena hasilnya akan digunakan oleh manajemen dalam pemilihan proses, perencanaan kapasitas, penjadwalan, dan pengendalian persediaan . 

UNIVERSITAS PAMULANG 

24 

Berdasarkan jangka waktunya, peramalan dapat dikategorikan menjadi tiga jenis, yaitu peramalan jangka pendek (short-term), jangka menengah (medium-term), dan jangka panjang (long-term) (Armstrong, 2001b). Sementara itu, berdasarkan metode pendekatannya, peramalan secara luas diklasifikasikan menjadi empat jenis utama. Pertama, metode kualitatif yang bersifat subjektif dan dipengaruhi oleh opini, seperti survei pasar, opini eksekutif, gabungan tenaga penjualan, dan metode _Delphi_ . Kedua, metode _time series_ yang menggunakan data historis untuk memproyeksikan masa depan dengan asumsi pola masa lalu akan berulang . Ketiga, metode kausal yang mengembangkan model sebab-akibat antara variabel yang diramalkan dengan variabel lain yang memengaruhinya, seperti analisis regresi . Keempat, metode simulasi yang mengombinasikan pendekatan kausal dan _time series_ untuk meniru perilaku konsumen dalam kondisi tertentu, misalnya simulasi Monte Carlo(Rob J. Hyndman & Athanasopoulos, 2018) . 

Dalam praktiknya, terdapat dua pendekatan utama dalam melakukan peramalan, yaitu _Top Down Forecasting_ dan _Bottom Up Forecasting_ . _Top Down Forecasting_ dimulai dengan hasil peramalan dari kondisi bisnis umum (makro) yang kemudian diterjemahkan ke dalam peramalan industri dan akhirnya ke pangsa pasar perusahaan. Metode statistik seperti analisis regresi dan korelasi sering digunakan dalam pendekatan ini. Sebaliknya, _Bottom Up Forecasting_ dimulai dengan membuat perkiraan permintaan untuk produk akhir individual, kemudian menjumlahkannya untuk mendapatkan ramalan agregat. Banyak perusahaan menggabungkan kedua metode ini untuk mendapatkan hasil peramalan yang lebih akurat . 

##### **2.2.2 Peramalan Penjualan (** **_Sales Forecasting_ )** 

Peramalan penjualan merupakan bagian spesifik dari peramalan yang berfokus pada estimasi permintaan produk atau jasa di masa mendatang, yang dinyatakan dalam kuantitas atau nilai penjualan sebagai fungsi dari waktu . Peramalan penjualan adalah dasar utama bagi perencanaan operasional dan strategis perusahaan. Tanpa estimasi penjualan yang akurat, perusahaan akan 

UNIVERSITAS PAMULANG 

25 

kesulitan dalam menentukan target produksi, mengelola stok, dan mengalokasikan sumber daya secara efektif. 

Tujuan utama dari peramalan penjualan adalah untuk meminimalkan risiko ketidakpastian di masa depan terkait penerimaan pendapatan. Dengan mengetahui proyeksi penjualan, perusahaan dapat merencanakan kebutuhan produksi, tenaga kerja, dan bahan baku secara lebih presisi. Ketidakakuratan dalam peramalan penjualan dapat berakibat fatal, seperti kelebihan produksi ( _overproduction_ ) yang menyebabkan pemborosan biaya penyimpanan dan modal, atau kekurangan produksi ( _underproduction_ ) yang mengakibatkan hilangnya peluang pendapatan dan ketidakpuasan pelanggan. Sebagai contoh, dalam penelitian di PT Seiwa Indonesia, ketidaksesuaian antara permintaan dan produksi mendorong dilakukannya analisis peramalan untuk menentukan metode terbaik guna memprediksi permintaan produk RBL, dengan tujuan mengurangi kerugian akibat produksi berlebih(Azizah & Nisah, 2024) . 

Peramalan penjualan yang baik juga menjadi landasan bagi berbagai fungsi bisnis lainnya. Di bidang keuangan, peramalan ini membantu dalam penyusunan anggaran dan proyeksi arus kas. Di bidang pemasaran, peramalan menjadi dasar untuk merancang strategi promosi dan penetapan harga. Di bidang manajemen rantai pasok, peramalan penjualan krusial untuk menentukan kebijakan pengadaan bahan baku, manajemen inventaris, dan logistik . Dengan data historis penjualan, perusahaan dapat menghitung estimasi hari stok tersisa ( _days of stock left_ ), yang memungkinkan pengambilan keputusan tepat waktu mengenai kapan harus memesan stok tambahan atau mengidentifikasi produk yang kurang diminati pasar . 

##### **2.2.3** **_Time Series Forecasting_** 

Time series forecasting atau peramalan deret waktu adalah metode peramalan kuantitatif yang menggunakan data historis untuk memprediksi nilai di masa depan. Metode ini didasarkan pada asumsi bahwa pola historis dari suatu data, seperti permintaan atau penjualan, merupakan indikator yang baik bagi 

UNIVERSITAS PAMULANG 

26 

pola di masa depan . Data deret waktu adalah serangkaian observasi yang diurutkan berdasarkan waktu, dan dalam analisisnya, model-model time series memungkinkan adanya korelasi antar observasi, di mana korelasi tersebut biasanya paling besar untuk titik-titik yang berdekatan dalam waktu(Rob J. Hyndman & Athanasopoulos, 2018) . 

Dalam analisis deret waktu, model time series berusaha untuk mengidentifikasi dan mengekstrak pola-pola bermakna yang tersembunyi di dalam data historis. Pola-pola ini tidak sekadar muncul secara acak, melainkan membentuk karakteristik tertentu yang dapat dikenali dan dipelajari. Salah satu pola yang paling mendasar adalah tren, yaitu gerakan data dalam jangka panjang yang menunjukkan kecenderungan naik atau turun secara konsisten. Selain itu, terdapat pula pola musiman, yang merupakan fluktuasi periodik yang terjadi secara teratur dalam interval waktu tertentu—misalnya mingguan, bulanan, atau tahunan—yang biasanya dipengaruhi oleh faktor kalender atau pergantian musim. 

Namun, tidak semua fluktuasi bersifat tetap dan dapat diprediksi. Dalam dunia ekonomi dan bisnis, misalnya, sering dijumpai pola siklus, yakni fluktuasi jangka panjang yang muncul akibat pengaruh kondisi ekonomi, namun periode kemunculannya tidak tetap seperti halnya musiman. Terakhir, ada pula variasi acak atau irregular variation, yaitu fluktuasi yang benar-benar tidak dapat diprediksi karena muncul akibat kejadian-kejadian tak terduga. Pola inilah yang kerap menjadi tantangan dalam pemodelan, karena tidak mengikuti pola tertentu dan sulit diantisipasi. Dengan memahami keempat pola ini secara menyeluruh, model time series dapat menyusun gambaran yang lebih utuh tentang perilaku data di masa lalu dan meramalkan kemungkinan pergerakannya di masa depan. 

Beberapa model time series yang umum digunakan antara lain _Autoregressive_ (AR), _Moving Average_ (MA), _Autoregressive Integrated Moving Average_ (ARIMA), dan model pemulusan ( _Exponential Smoothing_ ) (George E.P. Box, 2014). Model-model ini secara efektif dapat menangani korelasi antar waktu dan efek musiman. Kemajuan komputasi saat ini bahkan memungkinkan pemilihan parameter model secara otomatis, seperti 

UNIVERSITAS PAMULANG 

27 

yang diimplementasikan dalam paket " _forecast_ " di R, yang membantu pengguna untuk mendapatkan model time series yang optimal . 

##### **2.2.4 AIDA (** **_Attention, Interest, Desire, Action_ )** 

Konsep AIDA merupakan singkatan yang sudah lama dikenal dalam dunia pemasaran sebagai kerangka acuan untuk memahami empat tahapan yang dilalui konsumen dalam proses pembelian, yaitu _Attention_ , _Interest_ , _Desire_ , dan _Action_ . Model ini tergolong sederhana namun tetap relevan digunakan sebagai pedoman dalam menyusun strategi komunikasi pemasaran (Apriandi et al., 2023). 

Pada tahap _Attention_ , tugas utama pemasar adalah merancang media informasi yang mampu menarik perhatian konsumen sejak awal. Ini bisa dicapai melalui pernyataan, kata-kata, atau visual yang cukup kuat untuk membuat calon konsumen berhenti sejenak dan memperhatikan pesan yang disampaikan. Menurut Kotler & Armstrong (2018), daya tarik sebuah iklan pada tahap ini ditentukan oleh tiga hal: konten pesan itu sendiri, seberapa sering iklan ditayangkan, dan bagaimana visualisasinya disajikan. 

Setelah perhatian konsumen berhasil didapat, tahap berikutnya adalah _Interest_ — membangun ketertarikan agar target audiens bersedia meluangkan waktu untuk memahami pesan lebih jauh. Banyak media promosi gagal justru di tahap ini karena tidak berhasil membuat konsumen ingin tahu lebih lanjut. Untuk membangkitkan minat, pemasar sebaiknya menawarkan solusi atas masalah atau harapan konsumen, serta menjelaskan manfaat produk secara gamblang — bukan sekadar memaparkan fitur dan membiarkan konsumen menyimpulkan sendiri manfaatnya. Assael (2018) mendefinisikan _Interest_ sebagai munculnya ketertarikan konsumen terhadap objek yang diperkenalkan pemasar, yang dipengaruhi oleh efektivitas media, persepsi konsumen terhadap produk setelah melihat iklan, dan kejelasan pesan yang disampaikan. 

Tahap _Desire_ muncul ketika pemasar mulai membangkitkan keinginan konsumen untuk memiliki atau mencoba produk. Di titik ini, pemasar perlu jeli 

UNIVERSITAS PAMULANG 

28 

membaca kesiapan konsumen: apakah mereka sudah cukup termotivasi, sudah mulai goyah secara emosional, namun masih menyimpan keraguan — misalnya mempertanyakan apakah produk benar-benar akan memberikan manfaat seperti yang dijanjikan dalam iklan. Dengan kata lain, _Desire_ adalah proses mengubah ketertarikan menjadi keinginan nyata untuk memiliki dan merasakan produk tersebut. 

Tahap terakhir, _Action_ , adalah puncak dari keseluruhan proses — mendorong konsumen untuk benar-benar mengambil keputusan membeli. Pemasar perlu memandu target audiens secara eksplisit, karena konsumen cenderung baru bertindak setelah diberi arahan yang jelas, termasuk kadang informasi harga sebagai pemicu keputusan. Mengubah minat menjadi tindakan pembelian nyata bukan perkara mudah; diperlukan pemilihan kata yang tepat, bahkan kalimat perintah ( _call to action_ ) yang tegas agar konsumen benar-benar bergerak menuju keputusan pembelian 

##### **2.2.5 Metode Peramalan** **_Baseline_** 

Metode peramalan _baseline_ adalah metode dasar dan sederhana yang sering digunakan sebagai titik awal dalam melakukan peramalan. Metodemetode ini menjadi fondasi penting dan sering digunakan sebagai pembanding untuk mengevaluasi kinerja metode yang lebih kompleks. Dalam konteks penelitian ini, metode baseline akan menjadi model individu yang akan digabungkan dalam _ensemble_ (Rob J. Hyndman & Athanasopoulos, 2018). 

##### **2.2.5.1** **_Moving Average_** 

Metode _Moving Average_ (rata-rata bergerak) adalah teknik peramalan yang menghitung rata-rata dari sejumlah data historis terbaru untuk memprediksi nilai periode berikutnya. Metode ini efektif untuk menghaluskan fluktuasi acak dalam data dan mengidentifikasi tren. Asumsi dasarnya adalah bahwa permintaan di masa depan akan mendekati rata-rata permintaan beberapa periode terakhir . 

Terdapat beberapa variasi dari metode _Moving Average_ , antara lain: 

UNIVERSITAS PAMULANG 

29 

1. _Single Moving Average_ (SMA): Memberikan bobot yang sama kepada setiap data dalam periode yang digunakan. 

2. _Weighted Moving Average_ (WMA): Memberikan bobot yang berbeda kepada setiap data, biasanya data terbaru diberi bobot lebih besar. 

3. _Double Moving Average_ : Digunakan untuk data yang menunjukkan tren, dengan menghaluskan data hasil SMA sekali lagi. 

Keunggulan utama metode ini adalah kesederhanaannya dan kemudahan dalam implementasi. Namun, metode ini memiliki kelemahan dalam menangani data dengan pola musiman atau tren yang kuat, serta memerlukan penentuan panjang periode (jendela) yang optimal. 

##### **2.2.5.2** **_Exponential Smoothing_** 

_Exponential Smoothing_ (pemulusan eksponensial) adalah metode peramalan yang memberikan bobot secara eksponensial menurun pada data historis, di mana data yang lebih baru memiliki bobot yang lebih tinggi. Metode ini lebih responsif terhadap perubahan pola data dibandingkan Moving Average(Rob J. Hyndman & Athanasopoulos, 2018) . Terdapat beberapa jenis Exponential Smoothing: 

1. _Single Exponential Smoothing_ (SES): Cocok untuk data yang tidak memiliki tren atau musiman. Model ini hanya memiliki satu parameter pemulusan ( _alpha_ ). 

2. _Double Exponential Smoothing_ (DES): Digunakan untuk data yang mengandung tren. Metode ini memiliki dua parameter pemulusan, satu untuk level dan satu untuk tren. Sebuah penelitian menggunakan metode ini untuk meramalkan permintaan produk di PT Seiwa Indonesia (Azizah & Nisah, 2024). 

UNIVERSITAS PAMULANG 

30 

3. _Triple Exponential Smoothing_ (TES) atau HoltWinters: Digunakan untuk data yang mengandung tren dan musiman. 

Pemilihan parameter pemulusan ( _alpha, beta, gamma_ ) sangat penting dalam metode ini karena menentukan seberapa cepat model merespons perubahan data. 

##### **2.2.5.3** **_Recurrent Neural Network_** 

_Recurrent Neural Network_ (RNN) merupakan salah satu jenis jaringan saraf tiruan ( _Artificial Neural Network_ ) yang dirancang untuk mengolah data berurutan ( _sequential data_ ) atau data _time series_ . Berbeda dengan jaringan saraf konvensional, RNN memiliki mekanisme umpan balik ( _feedback loop_ ) yang memungkinkan jaringan menyimpan informasi dari proses sebelumnya ke proses berikutnya _._ 

RNN mampu mempertahankan informasi dari waktu sebelumnya melalui _hidden state_ (EL Mahjouby et al., 2024). Pada setiap langkah waktu, model menerima input baru dan mengombinasikannya dengan informasi yang telah dipelajari sebelumnya untuk menghasilkan output. Kemampuan ini membuat RNN efektif digunakan pada berbagai bidang seperti prediksi harga saham, peramalan mata uang, pengenalan suara, dan pemrosesan bahasa alami. 

Gambar 2.1 mengilustrasikan sebuah elemen jaringan saraf yang diberi label “A.” Elemen ini menerima masukan yang disebut “𝑋𝑡” dan menghasilkan nilai keluaran yang disebut “ℎ𝑡.” Keberadaan loop memungkinkan pertukaran informasi antar berbagai langkah dalam jaringan. Anda dapat membayangkan jaringan saraf berulang (RNN) memiliki banyak instance “A” di dalam jaringan, di mana setiap _instance_ secara berurutan mengirimkan informasi ke instance berikutnya. 

UNIVERSITAS PAMULANG 

31 



<!-- Start of picture text -->
) © ® © ©)<br>= LARPLAPRLAL AL<br>© © © © . @<br><!-- End of picture text -->

##### **2.2.6.2** **_Weighted Average Ensemble_** 

_Weighted Average Ensemble_ adalah salah satu bentuk penggabungan model yang paling intuitif (Adhikari & Agrawal, 2012). Dalam pendekatan ini, prediksi akhir dihasilkan dengan menghitung rata-rata tertimbang dari prediksi semua model individu. Setiap model diberikan bobot tertentu yang mencerminkan kontribusi atau tingkat kepercayaan terhadap model tersebut . Semakin tinggi bobot sebuah model, semakin besar pengaruhnya terhadap hasil prediksi akhir. 

Secara matematis, untuk K model dengan prediksi 𝑦̂𝑖 dan bobot 𝑤𝑖 untuk setiap model i, prediksi ensemble 𝑦̂𝑒𝑛𝑠𝑒𝑚𝑏𝑙𝑒 dihitung sebagai: 



dengan batasan bahwa jumlah seluruh bobot sama dengan 1 (∑𝐾𝑖=1 𝑤𝑖 = 1) . 

Pendekatan ini memberikan fleksibilitas untuk memberikan penekanan lebih pada model yang memiliki kinerja lebih baik. Sebagai contoh, jika model regresi linier terbukti paling akurat pada data historis, maka model tersebut dapat diberi bobot tertinggi, sementara model lain dengan akurasi lebih rendah diberi bobot lebih kecil. Penelitian menunjukkan bahwa dengan memberikan bobot yang tepat, _weighted average ensemble_ dapat menghasilkan peningkatan akurasi yang signifikan dibandingkan _simple average_ . 

##### **2.2.6.3 Metode Penentuan Bobot /** **_Weight_** 

Penentuan bobot ( _weight_ ) merupakan langkah krusial dalam _weighted average ensemble_ . Nilai bobot yang optimal tidak diketahui secara apriori dan harus ditentukan melalui suatu prosedur. Metode penentuan bobot dapat dikategorikan menjadi dua pendekatan utama: 

1. Metode _Heuristic_ atau Berbasis Kinerja: Bobot ditentukan berdasarkan metrik kinerja model pada data validasi. Misalnya, bobot dapat dihitung secara proporsional berdasarkan nilai akurasi atau kebalikan dari nilai kesalahan (error). Model dengan error terkecil mendapatkan bobot terbesar. 

UNIVERSITAS PAMULANG 

33 

2. Metode Optimasi: Bobot ditentukan dengan memperlakukan masalah penentuan bobot sebagai suatu masalah optimasi. Tujuannya adalah mencari kombinasi bobot yang meminimalkan fungsi kesalahan tertentu (misalnya, MSE atau RMSE) pada data validasi. Proses ini dapat diselesaikan dengan berbagai algoritma optimasi, termasuk algoritma metaheuristik seperti _Grey Wolf Optimizer_ (GWO) yang menjadi fokus dalam penelitian ini. 

##### **2.2.7 Algoritma Optimasi** 

##### **2.2.7.1 Permasalahan Optimasi** 

Masalah optimasi adalah suatu proses untuk mencari solusi terbaik dari sekumpulan solusi yang mungkin dalam suatu ruang pencarian, dengan tujuan memaksimalkan atau meminimalkan suatu fungsi tertentu yang disebut fungsi objektif (Mirjalili et al., 2014). Dalam konteks ilmiah dan teknik, masalah optimasi muncul di berbagai bidang, seperti desain teknik, penjadwalan, kontrol robot cerdas, perencanaan misi eksplorasi ruang angkasa, pemilihan kluster kepala pada jaringan sensor, dan pengembangan protokol keamanan _blockchain_ . 

Dalam sebuah masalah optimasi, terdapat tiga komponen utama yang saling terkait dan membentuk kerangka kerja dalam pencarian solusi terbaik. Komponen pertama adalah variabel keputusan, yaitu parameter-parameter yang dapat diubah atau disesuaikan oleh pengambil keputusan untuk menemukan solusi yang diinginkan. Variabel inilah yang menjadi titik kendali dalam proses optimasi, karena nilainya akan ditentukan melalui perhitungan atau algoritma tertentu. Selanjutnya ada fungsi objektif, yang merupakan representasi matematis dari tujuan yang ingin dicapai. Fungsi ini berfungsi sebagai alat ukur untuk menilai kualitas suatu solusi, apakah solusi tersebut semakin mendekati target yang diinginkan atau justru sebaliknya. Dalam praktiknya, fungsi objektif dapat berupa maksimisasi keuntungan, minimisasi biaya, atau capaian tertentu lainnya tergantung pada konteks permasalahan. Komponen ketiga adalah kendala atau constraints, yaitu batasan-batasan yang harus dipenuhi agar suatu solusi dapat dinyatakan layak. Kendala ini mencerminkan keterbatasan sumber 

UNIVERSITAS PAMULANG 

34 

daya, kebijakan, atau kondisi nyata yang tidak dapat diabaikan dalam proses pengambilan keputusan. Tanpa kendala, solusi yang dihasilkan mungkin saja optimal secara matematis, namun tidak dapat diterapkan dalam praktik karena melanggar batasan yang ada. Ketiga komponen ini bekerja secara simultan dalam membentuk masalah optimasi yang utuh dan realistis. 

Dalam penelitian ini, masalah optimasi yang dihadapi adalah menentukan bobot yang optimal untuk model _ensemble_ . Variabel keputusannya adalah nilai bobot untuk setiap model _baseline_ ( _Moving Average, Exponential Smoothing, Linear Regression_ ), fungsi objektifnya adalah meminimalkan nilai kesalahan peramalan (misalnya, MSE), dan kendalanya adalah jumlah bobot harus sama dengan 1 dan setiap bobot bernilai antara 0 dan 1. 

##### **2.2.7.2 Algoritma Optimasi yang terinspirasi dari alam** 

Algoritma metaheuristik terinspirasi alam ( _nature-inspired metaheuristics_ ) adalah kelas algoritma optimasi yang terinspirasi oleh fenomena alam, baik biologis, fisik, maupun perilaku sosial. Algoritma ini sangat populer karena kemampuannya dalam menyelesaikan masalah optimasi yang kompleks, non-linear, dan memiliki banyak variabel, di mana metode konvensional (seperti _gradient-based_ ) sulit diterapkan . 

Algoritma metaheuristik secara umum dikategorikan menjadi dua jenis utama berdasarkan jumlah solusi yang diproses : 

1. _Single-solution based_ : Algoritma yang hanya memproses satu solusi dalam setiap iterasi, seperti _Simulated Annealing_ dan _Iterated Local Search_ . 

2. _Population-based_ : Algoritma yang memproses sekumpulan solusi (populasi) dalam setiap iterasi. Algoritma ini umumnya lebih robust dalam menemukan solusi global karena melakukan pencarian secara simultan di berbagai wilayah ruang solusi. Contoh populer dalam kategori ini termasuk _Genetic Algorithms_ (terinspirasi evolusi), _Particle Swarm Optimization_ (terinspirasi perilaku kawanan burung atau ikan), 

UNIVERSITAS PAMULANG 

35 



<!-- Start of picture text -->
A.<br>A»<br>pC<br>FCN<br><!-- End of picture text -->

##### Gambar 2.2 Hirarki sosial serigala abu-abu (Sumber: (Mirjalili et al., 2014)) 

Salah satu aspek paling menarik dari perilaku serigala abu-abu adalah hierarki sosialnya yang ketat, dan dalam pemodelan _Grey Wolf Optimizer_ (GWO), hierarki ini diadaptasi menjadi empat tingkatan untuk keperluan optimasi. Di puncak hierarki terdapat _Alpha_ (α), pemimpin kelompok yang bertanggung jawab atas berbagai keputusan penting seperti menentukan tempat berburu atau waktu istirahat. Dalam algoritma GWO, posisi ini merepresentasikan solusi terbaik yang memiliki nilai fungsi objektif paling optimal. Setingkat di bawahnya ada _Beta_ (β), yang berperan sebagai penasihat dan pembantu bagi _Alpha_ . Mereka tidak hanya membantu dalam pengambilan keputusan, tetapi juga menjadi kandidat terkuat untuk menggantikan _Alpha_ jika diperlukan. _Beta_ tetap menghormati otoritas _Alpha_ , namun memiliki wewenang untuk memimpin serigala di tingkat bawahnya, dan dalam konteks algoritma, _Beta_ merepresentasikan solusi terbaik kedua. Selanjutnya ada _Delta (δ)_ yang menempati tingkatan ketiga dalam struktur sosial ini. _Delta_ tunduk kepada _Alpha_ dan _Beta_ , namun memiliki dominasi atas serigala di tingkat paling bawah. Kelompok ini terdiri dari beragam peran seperti serigala pengintai, penjaga, pemburu, hingga yang merawat anak-anak serigala. Dalam GWO, _Delta_ menjadi solusi terbaik ketiga yang ikut memandu proses pencarian. Dan yang terakhir adalah _Omega_ (ω), serigala dengan tingkatan terendah dalam hierarki. Mereka harus tunduk kepada semua serigala lainnya dan sering dianggap sebagai "kambing hitam" dalam kelompok. Namun demikian, keberadaan _Omega_ bukanlah tanpa arti, mereka justru berperan penting dalam menjaga kohesi dan kepuasan seluruh struktur sosial. Dalam implementasi algoritma, seluruh solusi yang tersisa dalam populasi akan dikategorikan sebagai _Omega_ . 

Selain hierarki sosial yang ketat, perilaku berburu secara berkelompok merupakan aspek menarik lainnya dari kehidupan serigala abu-abu. Menurut (Muro et al., 2011), tahapan utama dalam perburuan serigala abu-abu terdiri dari tiga fase penting. Fase pertama dimulai dengan melacak, mengejar, dan mendekati mangsa. Setelah itu, serigala-serigala tersebut memasuki fase kedua 

UNIVERSITAS PAMULANG 

37 



<!-- Start of picture text -->
“s .<br>A -- =~ 7cS gq z<br>|<br>|<br>4 ~ aa ‘ % re<br>'<br>S<br>s ‘ ? 4<br>L x| : ; as=~oa—) adave] p-Fe % j 5 ‘ ed, E sae SS =“ =<br>= oy, Pescie:RIKCeZ Ss Z ieem3C Oe a© sae| eess- \ ae ss ~<br>:<br>; 7 : :<br>fe “* Se — a<br><!-- End of picture text -->

~~<u>a</u>~~ 

Para serigala _omega_ akan mengikuti ketiga serigala pemandu tersebut dalam mencari solusi optimal. 

##### **a. Mengepung (** **_Encircling Prey_ )** 

Perilaku mengepung mangsa oleh serigala abu-abu direpresentasikan secara matematis melalui persamaan: 





di mana 𝑡 menunjukkan iterasi saat ini, 𝑋⃗𝑝 adalah vektor posisi mangsa, 𝑋⃗ adalah vektor posisi serigala abu-abu, 𝐴⃗ dan 𝐶⃗ adalah vektor koefisien yang dihitung sebagai: 





Nilai 𝑎⃗ menurun secara linier dari 2 ke 0 selama iterasi, sementara 𝑟⃗1 dan 𝑟⃗2 adalah vektor acak dalam interval [0,1]. Untuk melihat pengaruh dari persamaan (2.3) dan (2.4), vektor posisi dua dimensi beserta beberapa kemungkinan posisi diilustrasikan pada Gambar 2.4 (a). Seperti yang terlihat pada gambar tersebut, seekor serigala abu-abu yang berada pada posisi (X,Y) dapat memperbarui posisinya berdasarkan posisi mangsa (X* _,_ Y _*_ ). Berbagai tempat di sekitar agen terbaik dapat dicapai dari posisi saat ini dengan menyesuaikan nilai vektor 𝐴⃗ dan 𝐶⃗. Sebagai contoh, posisi (X*-X, Y) dapat dicapai dengan mengatur 𝐴⃗ = (1,0) dan 𝐶⃗ = (1,1). Kemungkinan posisi terbaru dari seekor serigala abu-abu dalam ruang tiga dimensi diperlihatkan pada Gambar 2.4 (b). Perlu dicatat bahwa vektor acak 𝐴⃗ dan 𝐶⃗ memungkinkan serigala untuk mencapai sembarang posisi di antara titik-titik yang diilustrasikan pada Gambar 2.4. Dengan demikian, seekor serigala abu-abu dapat memperbarui posisinya di dalam ruang di sekitar mangsa pada lokasi acak mana pun dengan menggunakan persamaan (2.3) dan (2.4). 

UNIVERSITAS PAMULANG 

39 



<!-- Start of picture text -->
~< XX > XYZ) (YD) oy XY.Z)<br>(XX) wey) Y) OFXYZY pg weyzp are) 2<br>~ P ; t F<br>7, Ga f (X*-XYZ*Z) YZ" ©) (XYZZ<br>Ti» tA r K*XYZ*-Z) (X*Y,.Z4.2) (X%Y,Z"2) arez)<br>Ny7>J<br>\H 7 = tk (X*, YY", 24)<br>OX) (XY) >, Mh re} PD yx zs)<br>(Tr fT oS Py) (X*XY"Z*2) | 7 IX*.Y*.2*- 2] op (X.Y*%Z*-Z)_ (XY*Y.Z)<br>UK 7) AY AS Paws - 1 . | ‘ .<br>« YZ ' \ Ae 4 J2. »<br>7 7 Hi \ N / / «<br>hdso > Ay A> J be /a4 DK > YZ)<br>pai “A Vor’y ’<br>> Rey OOK YRYZZ! ore. ye-y.2*-Z) (YeyZ2Z)<br>OXY) OP) wyey) (b)<br>(a)<br><!-- End of picture text -->

=>>> => > > =>>> 

>> > = > > > = 

~~<u>SS</u>~~ 

>>>= 

>> > 

> 



<!-- Start of picture text -->
s/f/ Z ’a“ _-- rTsy N\ \<br>o_/I,(te,\ ot7 ay \\4,1ot\\4ot 4 y y oe eooTrTTTe~. s S ‘\<br>V\ yo / a \<br>\ N .NY x\ NA~n Lee----p> -ard, / ‘ !HP</ | ‘/ . a 2 \\14| ON\ !|<br>\\ \ !H i!<br>\\\7\\ ¢// 7//<br>\ \ Ss --- as,<br>ee “<br>Dy<br><oo Mi @ “<br>¢ ¢“ ~ Ny ‘\ XN Movelove<br>/ fo 7een ~ . XV C) r<br>i‘ / ‘“\ \ \\<br>f // \\ \\ 6<br>! ! \ \<br>! \ \4 1\i \ a; /1!i F!}|Dy orother@any.<br>\ \ 7 4 hunters<br>\ \ CY3/ ;<br>\Sete 4 oe<br>\\ . “7 7 / C) Estimatedposition<br>a 7 of the<br>~. ~~ ee - prey<br><!-- End of picture text -->

### ~~<u>LE</u>~~ 



<!-- Start of picture text -->
MN<br>ee ><br>(a) (b)<br><!-- End of picture text -->

eksplorasi dan menghindari jebakan solusi lokal. Perlu dicatat bahwa tidak seperti A yang menurun secara linear, C sengaja dirancang untuk selalu memberikan nilai acak sepanjang iterasi. Dengan demikian, eksplorasi tetap berlangsung tidak hanya di awal, tetapi juga hingga iterasi akhir. Komponen ini sangat bermanfaat ketika terjadi stagnasi pada solusi lokal, terutama di akhir iterasi. 

Vektor C juga dapat dimaknai sebagai representasi hambatan alam yang ditemui serigala saat mendekati mangsa. Di alam liar, berbagai rintangan kerap menghalangi jalur perburuan serigala, membuat mereka tidak dapat mendekati mangsa dengan cepat dan mudah. Fungsinya persis seperti yang dilakukan vektor C. Tergantung pada posisi serigala, vektor ini dapat secara acak memberi bobot pada mangsa, sehingga mangsa menjadi lebih sulit dan lebih jauh untuk dijangkau, atau sebaliknya. 

Secara ringkas, proses pencarian dalam algoritma GWO dimulai dengan membentuk populasi acak serigala abu-abu yang merupakan calon solusi. Sepanjang iterasi berlangsung, serigala alpha, beta, dan delta akan memperkirakan posisi mangsa yang paling mungkin. Setiap calon solusi kemudian memperbarui jaraknya dari mangsa. Parameter a secara bertahap dikurangi dari 2 menjadi 0 untuk mengatur keseimbangan antara eksplorasi di awal dan eksploitasi di akhir. Calon solusi akan cenderung menjauh dari mangsa saat |A|>1 dan mendekat saat |A|<1. Algoritma GWO akan berakhir setelah kriteria penghentian terpenuhi. 

Berikut ini adalah _pseudo code_ algoritma GWO 

**`Inisialisasi`** `populasi serigala abu-abu` 𝑋𝑖 `(i = 1, 2, ..., n)` **`Inisialisasi`** `parameter a, A, dan C` **`Hitung`** `nilai` _`fitness`_ `untuk setiap agen pencarian` **`Tentukan`** `:` 𝑋𝛼 `= agen pencarian dengan` _`fitness`_ `terbaik` 𝑋𝛽 `= agen pencarian dengan` _`fitness`_ `terbaik kedua` 𝑋𝛿 `= agen pencarian dengan` _`fitness`_ `terbaik ketiga` 

UNIVERSITAS PAMULANG 

43 

**`While`** `(t < jumlah iterasi maksimum)` **`do For`** `setiap agen pencarian: Perbarui posisi agen pencarian menggunakan` **`Persamaan (2.12) End For`** `Perbarui nilai a, A, dan C Hitung nilai` _`fitness`_ `untuk semua agen pencarian Perbarui` 𝑋𝛼 `,` 𝑋𝛽 `, dan` 𝑋𝛿 `t = t + 1` **`End While Kembalikan`** 𝑋𝛼 `sebagai solusi terbaik` 

Gambar 2.7 _Pseudo Code_ Algoritma GWO 

##### **2.3 Kerangka Pemikiran** 

Dalam melakukan penelitian, penulis membuat sebuah kerangka pemikiran yang digunakan sebagai pedoman atau acuan penelitian ini. Agar mudah dipahami, kerangka pemikiran dapat diilustrasikan pada gambar berikut. 

UNIVERSITAS PAMULANG 

44 



<!-- Start of picture text -->
Permasalahan<br>Model Invidual seperti Moving Average, Exponential<br>Smoothing, dan Linear Regression memiliki akurasi<br>terbatas untuk data penjualan yang kKompleks.<br>Peluang<br>Model Ensemble dapat menggabungkan kelebihan<br>masing-masing model untuk mendapatkan akurasi<br>yang lebih baik.<br>Pendekatan<br>Weighted Ensemble dengan bobot yang dioptimasi<br>Metode<br>Grey Wolf Optimizer (GWO) menentukan bobot<br>optimal masing-masing model individu di dalam<br>model Ensemble<br>Pengujian<br>1.Menggunakan Dataset publik Store Sales<br>dari Kaggle<br>2. Evaluasi menggunakan MAE, MSE, RMSE,<br>MAPE, R2<br>3. Perbandingkan kinerja model ensemble<br>dengan model individu<br>Hasil<br>Weighted Ensemble berbobot yang optimal untuk<br>peramalan penjualan dengan akurasi yang lebih<br>baik<br><!-- End of picture text -->



<!-- Start of picture text -->
fn<br><!-- End of picture text -->

~~<mark>______________________f#f</mark>~~ 

konvensional seperti _Moving Average_ , _Exponential Smoothing_ , dan _Linear Regression_ memiliki akurasi yang terbatas ketika digunakan secara individu untuk meramalkan data penjualan yang bersifat kompleks, non-linear, dan dipengaruhi berbagai faktor eksternal. Keterbatasan ini disebabkan oleh karakteristik masing-masing model yang hanya mampu menangkap pola tertentu dari data, sehingga ketika dihadapkan pada pola data yang beragam dan dinamis, hasil peramalan menjadi kurang akurat dan berisiko menimbulkan kesalahan dalam pengambilan keputusan bisnis. 

Dari permasalahan tersebut terbuka peluang untuk memanfaatkan konsep _ensemble learning_ yang mampu menggabungkan kelebihan dari beberapa model sehingga dapat menghasilkan prediksi yang lebih akurat dan stabil dibandingkan model tunggal. _Ensemble learning_ menawarkan solusi dengan cara mengkombinasikan output dari beberapa model untuk saling menutupi kelemahan masing-masing, di mana model yang unggul dalam menangkap pola tren dapat melengkapi model yang unggul dalam menangani pola musiman, dan seterusnya. 

Pendekatan yang dipilih dalam penelitian ini adalah _weighted ensemble_ , di mana setiap model baseline diberi bobot kontribusi tertentu terhadap hasil akhir. Pendekatan ini memberikan fleksibilitas untuk memberikan penekanan lebih pada model yang memiliki kinerja lebih baik, namun bobot yang optimal tidak diketahui secara apriori dan harus dicari melalui suatu proses optimasi. Pencarian bobot secara manual atau berdasarkan aturan tetap tidak akan menghasilkan kombinasi yang paling optimal karena kompleksitas interaksi antar model. 

Metode yang digunakan untuk menentukan bobot optimal adalah algoritma _Grey Wolf Optimizer_ (GWO), yang terinspirasi dari hierarki sosial dan perilaku berburu serigala abu-abu. GWO akan mencari kombinasi bobot w₁, w₂, w₃ untuk model _Moving Average_ , _Exponential Smoothing_ , dan _Linear Regression_ yang meminimalkan nilai kesalahan prediksi pada data validasi. Algoritma ini dipilih karena kemampuannya dalam menyeimbangkan eksplorasi dan eksploitasi ruang pencarian, serta telah terbukti efektif dalam berbagai 

UNIVERSITAS PAMULANG 

46 

masalah optimasi di berbagai domain. 

Tahap pengujian dilakukan menggunakan dataset publik _Store Sales - Time Series Forecasting_ dari Kaggle yang memiliki karakteristik kompleks dengan jutaan baris data, mencakup 54 toko dan 33 kategori produk selama periode 2013 hingga 2017. Kinerja model _ensemble_ yang dioptimasi GWO akan dievaluasi menggunakan metrik MAE, MSE, RMSE, MAPE, dan R², serta dibandingkan dengan kinerja masing-masing model baseline individu untuk mengukur seberapa besar peningkatan akurasi yang dicapai. 

Penelitian ini diharapkan menghasilkan hasil berupa model _ensemble_ optimal yang siap digunakan untuk peramalan penjualan dengan akurasi lebih baik dibandingkan metode konvensional. Model ini dapat menjadi alternatif solusi bagi perusahaan dalam mendukung pengambilan keputusan strategis terkait perencanaan produksi, pengelolaan persediaan, dan penyusunan strategi pemasaran, sehingga dapat meminimalkan risiko kerugian akibat kelebihan atau kekurangan stok 

UNIVERSITAS PAMULANG 

47 

#### **BAB III** 

#### **METODOLOGI** 

##### **3.1 Analisis Kebutuhan** 

Pada analisis kebutuhan akan dijelaskan tahapan yang dilakukan untuk menentukan kebutuhan-kebutuhan dalam melakukan penelitian. Analisis kebutuhan meliputi analisis kebutuhan data, kebutuhan perangkat keras, dan kebutuhan perangkat lunak. 

##### **3.1.1 Analisis Kebutuhan Data** 

Data yang digunakan dalam penelitian ini adalah dataset time series penjualan ritel dari Kaggle "Store Sales - Time Series Forecasting". Dataset ini menyediakan 3 juta baris data penjualan harian yang dijual di jaringan toko yang berlokasi di Ekuador dengan spesifikasi sebagai berikut: 

Tabel 3.1 Data Penjualan 

|**No**|<br>**Nama Atribut**|**Tipe Data**|**Deskripsi**|**Jenis**|
|---|---|---|---|---|
|1|date|Tanggal|Tanggal transaksi (2013-01-01 s.d.<br>2017-08-15)|<br>Fitur|
|2|store_nbr|Integer|Nomor identifikasi toko (1-54)|Fitur|
|3|Family|Kategori|Keluarga produk (33 kategori, misal:<br>_produce, dairy, beverages_)|<br>Fitur|
|4|Sales|Desimal|Total penjualan untuk setiap keluarga<br>produk pada suatu toko di tanggal<br>tertentu|<br> <br>Target|
|5|onpromotion|Integer|Jumlah<br>item<br>yang<br>sedang<br>dipromosikan|<br>Fitur|



UNIVERSITAS PAMULANG 

48 

##### **3.1.2 Analisis Kebutuhan Perangkat Keras** 

Pada penelitian ini dilakukan eksperimen dengan menggunakan komputer laptop untuk melakukan proses simulasi dan visualisasi terhadap model yang diusulkan. Spesifikasi perangkat keras yang digunakan pada penelitian ini adalah sebagai berikut: 

Tabel  3.2 Spesifikasi Perangkat Keras 

|**No.**||**Nama**|**Jenis**|
|---|---|---|---|
|1|_Processor_||Intel Core i7|
|2|RAM||16 GB|
|3|_Harddisk_||256 GB|



##### **3.1.3 Analisis Kebutuhan Perangkat Lunak** 

Perangkat lunak yang dibutuhkan untuk penelitian ini adalah sebagai berikut: 

Tabel  3.3 Spesifikasi Perangkat Lunak 

|**No.**|**Nama**|**Spesifikasi**|**Kegunaan**|
|---|---|---|---|
|1|Python|Versi terbaru|Bahasa pemrograman utama|
|2|Jupyter Notebook|6.4.0+|IDE untuk pengembangan<br>kode|
|3|NumPy|1.21.0+|Komputasi<br>numerik<br>dan<br>array|
|4|Pandas|1.3.0+|Manipulasi dan analisis data|
|5|Scikit-learn|0.24.0+|Implementasi<br>model<br>baseline|
|6|Matplotlib|3.4.0+|Visualisasi data|
|7|Seaborn|0.11.0+|Visualisasi statistik|
|8|Statsmodels|0.12.0+|Analisis time series|



UNIVERSITAS PAMULANG 

49 



<!-- Start of picture text -->
(ws) Studi Literatur Pengumpulan Data Pra Pemrosesan Pembersihan Data<br>Data Training 80% Pembagian Data<br>Implementasi Model Baseline _<br>(Moving Average, Exponential Data Testing 20%<br>Smoothing, Linear Regression)<br>Inisialisasi Populasi Hitung Fitness Tentukan pn. Beta, Update Posisi<br>idak’<br>Optimasi dengan Grey Wolf Optimizer<br>a<br>Kesimpulan Perbandingan Kinerja Ensemble dengan<br>MAE, MSE, R?RMSE, MAPE, bbot optimalP<br><!-- End of picture text -->

~~<u>a</u>~~ 

Setelah studi literatur selesai, peneliti mengumpulkan data yang akan digunakan dalam penelitian ini dari sumber yang valid, yaitu dataset _time series_ penjualan ritel dari kompetisi Kaggle "Store Sales - Time Series Forecasting", dengan mengunduh seluruh file yang diperlukan, memahami struktur dan deskripsi setiap atribut, serta memverifikasi kelengkapan data. Data mentah yang telah dikumpulkan kemudian memasuki tahap pra-pemrosesan data agar siap digunakan untuk pemodelan, yang terdiri dari tiga sub-tahapan utama yaitu pembersihan data, transformasi data, dan pembagian data. Pada sub-tahap pembersihan data, data dibersihkan dari berbagai masalah seperti nilai yang hilang, data duplikat, dan _outlier_ yang dapat mengganggu kinerja model, dengan melakukan proses pembersihan secara hati-hati agar tidak menghilangkan informasi penting. Selanjutnya pada sub-tahap transformasi data, data yang telah bersih ditransformasikan ke dalam format yang sesuai untuk pemodelan, seperti mengonversi tipe data, melakukan normalisasi jika diperlukan, dan mengintegrasikan data pendukung seperti informasi hari libur dan harga minyak untuk memaksimalkan informasi yang dapat diekstrak oleh model. 

Data yang telah melalui proses pembersihan dan transformasi kemudian memasuki sub-tahap pembagian data, di mana data dibagi menjadi dua bagian dengan proporsi yang telah ditentukan secara kronologis untuk menjaga urutan waktu dan menghindari kebocoran data. Sebanyak 80 persen dari total data dialokasikan sebagai data latih yang akan digunakan untuk melatih ketiga model baseline, mencakup periode waktu yang lebih panjang agar model dapat mempelajari pola historis dengan baik, sementara sisanya sebanyak 20 persen dialokasikan sebagai data uji yang akan digunakan untuk mengevaluasi kinerja model setelah proses pelatihan selesai, di mana data uji ini tidak pernah dilihat oleh model selama proses pelatihan sehingga mampu mengukur kemampuan generalisasi model secara objektif. Pada tahap implementasi model baseline, ketiga model peramalan dasar diimplementasikan menggunakan data latih yang telah disiapkan, mencakup pemilihan parameter awal, pelatihan model, dan validasi untuk memastikan model bekerja dengan baik. Model pertama yang diimplementasikan adalah _Moving Average_ yang bekerja dengan menghitung rata-rata dari sejumlah data historis terbaru untuk memprediksi nilai periode 

UNIVERSITAS PAMULANG 

51 

berikutnya dan efektif untuk menghaluskan fluktuasi acak serta mengidentifikasi tren jangka pendek. Model kedua adalah _Exponential Smoothing_ yang memberikan bobot secara eksponensial menurun pada data historis sehingga data yang lebih baru memiliki pengaruh lebih besar, menjadikannya lebih responsif terhadap perubahan pola data dibandingkan _Moving Average_ . Model ketiga adalah _Linear Regression_ yang memodelkan hubungan linier antara waktu sebagai variabel independen dan penjualan sebagai variabel dependen dengan mencari garis lurus terbaik yang dapat memprediksi nilai penjualan berdasarkan waktu. 

Setelah ketiga model baseline dilatih, masing-masing model menghasilkan prediksi untuk periode yang sama pada data uji, di mana prediksi dari ketiga model ini akan menjadi input utama untuk tahap optimasi selanjutnya. Tahap optimasi bobot dengan GWO merupakan inti dari penelitian, di mana algoritma _Grey Wolf Optimizer_ digunakan untuk mencari kombinasi bobot optimal bagi ketiga model baseline secara iteratif untuk meminimalkan nilai kesalahan prediksi dari model _ensemble_ . Proses GWO dimulai dengan inisialisasi populasi, yaitu membangkitkan populasi awal serigala secara acak di mana setiap serigala merepresentasikan satu kandidat solusi bobot dengan jumlah populasi ditentukan sebelumnya untuk menyeimbangkan antara kecepatan komputasi dan kualitas pencarian. Selanjutnya dilakukan perhitungan _fitness_ di mana setiap kandidat solusi dievaluasi nilai _fitness_ -nya menggunakan fungsi objektif yaitu nilai MAPE yang dihasilkan dari kombinasi bobot tersebut pada data validasi, dengan semakin kecil nilai MAPE berarti semakin baik kualitas kandidat solusi tersebut. Berdasarkan nilai _fitness_ , tiga kandidat solusi terbaik diidentifikasi sebagai Alpha, Beta, dan Delta yang akan memandu pencarian solusi selanjutnya, di mana Alpha merupakan solusi terbaik, diikuti Beta dan Delta sebagai solusi terbaik kedua dan ketiga. Posisi seluruh serigala dalam populasi kemudian diperbarui berdasarkan posisi Alpha, Beta, dan Delta menggunakan mekanisme matematis yang meniru perilaku berburu serigala, yang memungkinkan populasi untuk bergerak menuju wilayah solusi yang lebih baik. Setelah pembaruan posisi, dilakukan pengecekan apakah proses optimasi telah mencapai kondisi konvergen, yaitu ketika nilai _fitness_ tidak 

UNIVERSITAS PAMULANG 

52 

berubah signifikan atau jumlah iterasi maksimum telah tercapai, dan jika belum konvergen maka proses akan kembali ke tahap menghitung _fitness_ . Jika kondisi konvergen telah tercapai, posisi Alpha pada iterasi terakhir diambil sebagai bobot optimal yang terdiri dari tiga nilai yang masing-masing merepresentasikan kontribusi optimal dari model _Moving Average_ , _Exponential Smoothing_ , dan _Linear Regression_ . 

Dengan diperolehnya bobot optimal, model _weighted ensemble_ final dibangun dengan mengkombinasikan prediksi dari ketiga model baseline, di mana prediksi akhir dihasilkan sebagai jumlah terbobot dari prediksi masingmasing model. Model _ensemble_ yang telah terbentuk kemudian dievaluasi kinerjanya menggunakan data uji yang telah disiapkan sebelumnya dengan menghitung beberapa metrik kesalahan standar untuk peramalan. Metrik pertama yang dihitung adalah _Mean Absolute Error_ yang mengukur rata-rata absolut kesalahan prediksi dan memberikan gambaran seberapa besar kesalahan dalam unit yang sama dengan data asli. Metrik kedua adalah _Mean Squared Error_ yang mengukur rata-rata kuadrat kesalahan dan memberikan bobot lebih besar pada kesalahan yang besar serta sensitif terhadap keberadaan _outlier_ dalam data. Metrik ketiga adalah _Root Mean Squared Error_ yang merupakan akar kuadrat dari MSE dan mengembalikan satuan kesalahan ke satuan asli data sehingga lebih mudah diinterpretasikan. Metrik keempat adalah _Mean Absolute Percentage Error_ yang mengukur persentase kesalahan rata-rata dan sangat intuitif karena dinyatakan dalam persen serta memudahkan perbandingan antar data dengan skala yang berbeda. 

Setelah semua metrik evaluasi dihitung, dilakukan perbandingan kinerja antara model _ensemble_ yang dioptimasi GWO dengan kinerja masing-masing model baseline individu untuk mengukur seberapa besar peningkatan akurasi yang dicapai oleh model _ensemble_ . Berdasarkan hasil perbandingan kinerja tersebut, ditarik kesimpulan mengenai efektivitas model _ensemble_ yang dioptimasi dengan GWO dalam meningkatkan akurasi peramalan penjualan, yang juga mencakup analisis kelebihan dan keterbatasan dari metode yang diusulkan. Tahap akhir penelitian ditandai dengan selesainya seluruh rangkaian kegiatan dan diperolehnya model _ensemble_ optimal yang siap digunakan untuk 

UNIVERSITAS PAMULANG 

53 

peramalan penjualan, di mana hasil penelitian kemudian didokumentasikan dalam laporan. 

##### **3.3 Teknik Analisis** 

Analisis ini diajukan untuk membandingkan kinerja model _baseline_ individu, model _ensemble_ dengan bobot statis, dan model _ensemble_ dengan bobot optimal hasil optimasi GWO.Ukuran kinerja yang akan dianalisis meliputi _Mean Absolute Error_ (MAE), _Mean Squared Error_ (MSE), _Root Mean Squared Error_ (RMSE), dan _R-squared_ (R²). 

##### **3.3.1** **_Mean Absolute Error_ (MAE)** 

MAE adalah suatu metrik evaluasi kinerja yang umum digunakan dalam statistika dan pembelajaran mesin untuk mengukur seberapa dekat prediksi suatu model dengan nilai sebenarnya ( _ground truth_ ). MAE diukur sebagai rata-rata dari selisih absolut antara prediksi dan nilai sebenarnya. Persamaan berikut menunjukkan perhitungan MAE: 



Keterangan: 

𝑛  : Jumlah titik data. 

𝑦𝑖  : Nilai prediksi untuk titik data ke-i. 

𝑦𝑖  : Nilai aktual (sebenarnya) untuk titik data ke-i. 

##### **3.3.2** **_Mean Squared Error_ (MSE)** 

MSE adalah salah satu metrik evaluasi yang sering digunakan untuk mengukur performa model regresi. MSE menghitung rata-rata dari selisih kuadrat antara nilai aktual yang dihasilkan oleh model. MSE digunakan untuk menunjukkan seberapa besar kesalahan atau deviasi rata-rata antara prediksi model dengan nilai sebenarnya. Semakin kecil nilai MSE, semakin baik kinerja model, karena menunjukkan bahwa nilai prediksi mendekati nilai aktual. Persamaan perhitungan MSE: 

UNIVERSITAS PAMULANG 

54 



Keterangan: 

𝑛  : Jumlah titik data. 

𝑦𝑖  : Nilai prediksi untuk titik data ke-i. 

𝑦𝑖  : Nilai aktual (sebenarnya) untuk titik data ke-i. 

##### **3.3.3** **_Root Mean Squared Error_ (RMSE)** 

RMSE mengukur besarnya rata-rata kesalahan antara nilai yang diprediksi oleh model dan nilai aktual. Ini adalah standar deviasi dari residu (kesalahan prediksi). Karena setiap kesalahan dikuadratkan sebelum dirataratakan. Semakin rendah nilai RMSE, semakin baik model tersebut. Persamaan dari RMSE adalah sebagai berikut: 



Keterangan: 

𝑛  : Jumlah titik data. 

𝑦𝑖  : Nilai prediksi untuk titik data ke-i. 

𝑦𝑖  : Nilai aktual (sebenarnya) untuk titik data ke-i. 

- (𝑦𝑖̂ −𝑦𝑖) : Residu atau kesalahan prediksi. 

##### **3.3.4** **_Mean Absolute Percentage Error_ (MAPE)** 

_Mean Absolute Percentage Error_ (MAPE) merupakan salah satu ukuran statistik untuk mengevaluasi akurasi atau kinerja suatu model peramalan (forecasting) atau prediksi dengan mengukur seberapa besar kesalahan prediksi suatu model jika dibandingkan dengan nilai aktualnya, yang dinyatakan dalam bentuk persentase. Persamaan dari MAPE adalah sebagai berikut: 

UNIVERSITAS PAMULANG 

55 



Keterangan: 

- 𝑛 = jumlah data atau periode yang digunakan dalam evaluasi 

- 𝐴𝑡 = nilai aktual pada periode ke-𝑡 

- 𝐹𝑡 = nilai hasil prediksi atau peramalan pada periode ke-𝑡 

- ∣⋅∣ = nilai mutlak (absolut) yang memastikan kesalahan positif dan negatif tidak saling menghilangkan 

##### **3.3.5** **_R-squared_ (R²)** 

R<sup>2</sup> ( _R-squared_ ) atau Koefisien Determinasi adalah metrik yang menunjukkan proporsi varians dalam variabel dependen (variabel target) yang dapat diprediksi dari variabel independen (fitur) oleh model regresi. R<sup>2</sup> menggambarkan seberapa baik model regresi fit (cocok) dengan data. Nilai R<sup>2</sup> berkisar antara 0 hingga 1 (atau 0% hingga 100%), dimana nilai 0 berarti model tidak menjelaskan variabilitas respons sama sekali di sekitar rata-ratanya. Sedangkan nilai 1 (atau 100%) berarti model menjelaskan semua variabilitas data respons di sekitar rata-ratanya. Persamaan dari R<sup>2</sup> adalah sebagai berikut: 



Keterangan: 

𝑛  : Jumlah titik data. 

∑𝑛𝑖=1(𝑦𝑖̂ −𝑦𝑖)<sup>2</sup> : Jumlah kuadrat perbedaan antara nilai prediksi dan nilai aktual (kesalahan model). ∑𝑛𝑖=1(𝑦𝑖 −𝑦̅)<sup>2</sup> : Jumlah kuadrat perbedaan antara nilai aktual dan ratarata nilai aktual (variabilitas total dalam data). 

𝑦̅: Rata-rata dari nilai aktual 𝑦𝑖 

UNIVERSITAS PAMULANG 

56 

#### **HASIL DAN PEMBAHASAN** 

##### **4.1 Hasil** 

Hasil penelitian ini bertujuan untuk menerapkan _Grey Wolf Optimizer (GWO)_ untuk mengoptimasi model ensemble yang merupakan gabungan dari model MA, ES, dan RNN dalam peramalan penjualan. Beberapa tahapan mulai dari pengumpulan data, pra pemrosesan data, pembersihan data, pembagian data menjadi data training dan data testing, pembangunan model _baseline_ , hingga optimasi dengan menggunakan GWO untuk mencapai hasil peramalan penjualan yang lebih akurat. 

##### **4.1.1 Deskripsi Data** 

Dataset yang digunakan dalam penelitian ini adalah dataset time series penjualan ritel dari Kaggle "Store Sales - Time Series Forecasting" dengan karakteristik sebagaimana dijelaskan pada tabel 4.1 

Tabel 4.1. Karateristik Data 

|**No**|**Nama Atribut**|**Deskripsi**|
|---|---|---|
|1|date|Tanggal transaksi (2013-01-01 s.d. 2017-08-15)|
|2|store_nbr|Nomor identifikasi toko (1-54)|
|3|Family|Kategori produk (33 kategori, misal:_produce, dairy,_<br>_beverages_)|
|4|Sales|Total penjualan untuk setiap keluarga produk pada suatu<br>toko di tanggal tertentu (dalam unit)|
|5|onpromotion|Jumlah item yang sedang dipromosikan dalam kategori<br>produk tertentu, toko tertentu, dan tanggal tertentu.|



Data tersebut di atas memuat data historis penjualan retail berjumlah 3.000.088 _record_ di 54 toko sepanjang tahun 2013 sampai dengan sebagian 2017. Analisis statistik deskriptif dilakukan pada empat variabel utama, yaitu id, store_nbr, sales, dan onpromotion, dengan hasil sebagai berikut: 

UNIVERSITAS PAMULANG 

57 

Tabel 4.2. Analisa Deskriptif Dataset Awal 

||id|store_nbr|sales|onpromotion|
|---|---|---|---|---|
|count|3,000,888|3,000,888|3,000,888|3,000,888|
|mean|1,500,444|28|358|3|
|std|866,282|16|1,102|12|
|min|0|1|0|0|
|25%|750,222|14|0|0|
|50%|1,500,444|28|11|0|
|75%|2,250,665|41|196|0|
|max|3,000,887|54|124,717|741|



Dari analisa deskriptif dataset awal (sebelum pra pemrosesan), didapatkan bahwa untuk melakukan analisa _univariate timeseries_ , kita hanya memerlukan atribut sales saja.  Lebih jauh diberikan pada tabel 4.3 dan  tabel 4.4 10 baris teratas dan 10 baris terbawah dataset awal. 

Tabel 4.3. 10 Baris Teratas Dataset Awal 

|**id**|**date**|**store_nbr**|**family**|**sales**|**onpromotion**|
|---|---|---|---|---|---|
|**0**|2013-01-01|1|AUTOMOTIVE|0,0|0|
|**1**|2013-01-01|1|BABY CARE|0,0|0|
|**2**|2013-01-01|1|BEAUTY|0,0|0|
|**3**|2013-01-01|1|BEVERAGES|0,0|0|
|**4**|2013-01-01|1|BOOKS|0,0|0|
|**5**|2013-01-01|1|BREAD/BAKERY|0,0|0|
|**6**|2013-01-01|1|CELEBRATION|0,0|0|
|**7**|2013-01-01|1|CLEANING|0,0|0|
|**8**|2013-01-01|1|DAIRY|0,0|0|
|**9**|2013-01-01|1|DELI|0,0|0|



UNIVERSITAS PAMULANG 

58 

Tabel 4.4. 10 Baris Terbawah Dataset Awal 

|**id**|**date**|**store_**<br>**nbr**|**family**|**sales**|**onpromotion**|
|---|---|---|---|---|---|
|**3000878**|2017-08-15|9|MAGAZINES|11000|0|
|**3000879**|2017-08-15|9|MEATS|449228|0|
|**3000880**|2017-08-15|9|PERSONAL CARE|522000|11|
|**3000881**|2017-08-15|9|PET SUPPLIES|6000|0|
|**3000882**|2017-08-15|9|PLAYERS AND<br>ELECTRONICS|6000|0|
|**3000883**|2017-08-15|9|POULTRY|438133|0|
|**3000884**|2017-08-15|9|PREPARED FOODS|154553|1|
|**3000885**|2017-08-15|9|PRODUCE|241972<br>9|148|
|**3000886**|2017-08-15|9|SCHOOL AND OFFICE<br>SUPPLIES|121000|8|
|**3000887**|2017-08-15|9|SEAFOOD|16000|0|



##### **4.1.2 Pra Pemrosesan Data** 

Tahap pra-pemrosesan data dimulai dengan pembersihan data hilang melalui penghapusan baris yang memiliki nilai kosong (NaN atau null) pada kolom tanggal maupun kolom target (sales). Kemudian dilanjutkan dengan penghapusan baris data yang terduplikasi secara keseluruhan menggunakan untuk memastikan tidak ada redundansi data. Setelah data bersih, kolom tanggal dikonversi menjadi objek datetime agar urutan waktu dapat dipahami secara tepat oleh sistem. Data tersebut kemudian diurutkan secara kronologis dari yang terlama hingga yang terbaru . Tahap selanjutnya adalah agregasi data harian, di mana data dikelompokkan berdasarkan tanggal, sehingga transaksi yang terjadi pada hari yang sama disatukan menjadi satu baris total penjualan per hari. Untuk menjaga kualitas data, penanganan pencilan (outliers) diterapkan menggunakan metode _Interquartile Range_ (IQR). Sebagai langkah terakhir, nilai penjualan / _sales_ dinormalisasi ke dalam rentang [0, 1] menggunakan metode 

UNIVERSITAS PAMULANG 

59 

_MinMaxScaler_ . Pada Tabel 4.5 diberikan analisa deskriptif dataset hasil pra pemrosesan. 

Tabel 4.5. Analisa Deskriptif Dataset Hasil Pra Pemrosesan 

|**Statistik**|**date**|**sales**|
|---|---|---|
|**Count**|1.684|1.684|
|**Mean**|2015-04-24|0,488963|
|**Min**|2013-01-01|0,000000|
|**25%**|2014-02-26|0,339063|
|**50%**|2015-04-24|0,485007|
|**75%**|2016-06-19|0,603438|
|**max**|2017-08-15|1,000000|
|**std**|NaN|0,180004|



Dari dataset dengan record berjumlah 3.000.888 record dan 4 atribut, dihasilkan dataset hasil pra pemrosesan dengan jumlah record 1.684 record dengan 2 atribut, dengan atribut sales yang sudah ternormalisasi dengan skala data 0 - 1. Pada tabel 4.6 dan tabel 4.7 diberikan 10 baris teratas dan 10 baris terbawah dataset hasi pra pemrosesan. 

Tabel 4.6. 10 Baris Teratas Dataset Hasil Pra Pemrosesan 

|**Indeks**||**date**|**sales**|
|---|---|---|---|
||**0**|2013-01-01|0,000000|
||**1**|2013-01-02|0,380179|
||**2**|2013-01-03|0,276480|
||**3**|2013-01-04|0,271087|
||**4**|2013-01-05|0,365743|
||**5**|2013-01-06|0,398359|
||**6**|2013-01-07|0,256963|
||**7**|2013-01-08|0,243272|
||**8**|2013-01-09|0,231089|
||**9**|2013-01-10|0,197546|



UNIVERSITAS PAMULANG 

60 

Tabel 4.7. 10 Baris Terbawah Dataset Hasil Pra Pemrosesan 

|**Indeks**||**date**|**sales**|
|---|---|---|---|
||**1674**|2017-08-06|0,806485|
||**1675**|2017-08-07|0,612310|
||**1676**|2017-08-08|0,550923|
||**1677**|2017-08-09|0,563534|
||**1678**|2017-08-10|0,499794|
||**1679**|2017-08-11|0,634577|
||**1680**|2017-08-12|0,608587|
||**1681**|2017-08-13|0,664822|
||**1682**|2017-08-14|0,584164|
||**1683**|2017-08-15|0,585503|



##### **4.1.3 Pembagian Data Latih dan Data Uji** 

Sebelum membangun dan menguji model _baseline_ , langkah penting yang harus dilakukan adalah membagi dataset menjadi data latih dan data uji. Pembagian ini bertujuan agar model dapat dilatih pada sebagian data, kemudian dievaluasi performanya secara objektif pada data yang belum pernah dilihat sebelumnya. Dengan cara ini, keandalan dan kemampuan generalisasi model dalam mendeteksi pola-pola baru dapat diukur secara lebih akurat sebelum diterapkan pada data nyata di lapangan. Berikut adalah hasil dari pembagian data latih dan data uji. 

UNIVERSITAS PAMULANG 

61 



<!-- Start of picture text -->
Visualisasi Split Data (Train vs Test)<br>1.0 | —— Data Latih (Train: 1347 sampel) !\<br>—— Data Uji (Test: 337 sampel) 1<br>--- Batas Split Data |<br>'<br>! |<br>!<br>08 t | |<br>0.6 | | | '<br>Py= |<br>&<br>0.4 |\''<br>!<br>i<br>\'<br>1<br>0.2 1<br>1<br>1<br>'<br>1<br>i<br>0.0 '<br>2013 2014 2015 2016 2017<br>date<br><!-- End of picture text -->

## ~~<u>a</u>~~ 



<!-- Start of picture text -->
Kurva Training Loss RNN<br>0.035 —— Training Loss<br>0.030 4.<br>0.025 4 : :<br>w<br>uw<br>= 0.020<br>wv<br>nw<br>bar<br>0.015<br>0.010<br>0.005<br>o 20 40 60 80 100<br>Epoch<br><!-- End of picture text -->

~~<u>a</u>~~ 



<!-- Start of picture text -->
Prediksi MA vs Data Uji<br>1.0<br>0.8 I | i HI | MI ANIA|<br>06 Whi WW TA TUWUAWUA AW M TAOS VU TU<br>|<br>0.4<br>0.2<br>— Data Uji<br>~~ Prediksi<br>0.04<br>2016-09 2016-11 2017-01 2017-03 2017-05 2017-07<br>date<br><!-- End of picture text -->



<!-- Start of picture text -->
Prediksi ES vs Data Uji<br>1.0<br>0.8<br>0.6 LAME<br>An all | WA ae NMI |<br>e<br>0.4<br>0.2<br>— Data Uji<br>4.|——  Prediksi<br>0.0<br>2016-09 2016-11 2017-01 2017-03 2017-05 2017-07<br>date<br><!-- End of picture text -->

# | a ali 

|**RMSE**|0,1414|0,1352|0,1896|
|---|---|---|---|
|**R**<sup>**2**</sup>|-0,0089|0,0783|-0,8128|



Berdasarkan data pada Tabel 4.8, model RNN mencatatkan performa terbaik dengan nilai MAPE terendah, yakni 41,3504%. Hal ini menunjukkan bahwa arsitektur RNN memiliki kemampuan yang lebih baik dalam menangkap pola sekuensial dan tren non-linear harian, sehingga mampu menghasilkan MAPE yang lebih kecil dibandingkan dua model lainnya (MA sebesar 44,46% dan ES sebesar 46,41%). 

Meskipun ES memiliki nilai MAPE terburuk, model ini memiliki nilai RMSE yang terendah (0,1352), mengungguli MA (0,1414) dan RNN (0,1896). Tidak hanya itu,  ES mempunyai nilai koefisien determinasi (R2) satu-satunya yang positif (0,0783). Sebaliknya, model MA (-0,0089) dan Simple RNN (- 0,8128) menghasilkan nilai negatif. Nilai R2 negatif pada RNN dan MA menunjukkan bahwa meskipun persentase kesalahan (MAPE) lebih rendah dari ES,  kedua model ini tidak stabil dan kerap menghasilkan tebakan prediksi yang meleset pada titik-titik waktu tertentu, sehingga memperburuk tingkat varians secara keseluruhan 

##### **4.1.5 Hasil Optimasi Bobot Ensemble** 

Pada penelitian ini, algoritma _Grey Wolf Optimizer_ (GWO) digunakan untuk mencari kombinasi bobot optimal dengan fungsi tujuan meminimalkan nilai _Mean Absolute Percentage Error_ (MAPE). Mengingat GWO termasuk dalam kategori algoritma metaheuristik, pengujian tunggal tidak cukup untuk membuktikan keandalan solusi yang dihasilkan. Oleh karena itu dilakukan pengujian berulang sebanyak 30 kali ( _multi-run_ ) dengan parameter ukuran populasi serigala ( _n-wolf_ ) sebesar 20 dan jumlah iterasi maksimum sebesar 100 untuk setiap _run_ . Rekapitulasi hasil uji statistik dari 30 _runs_ tersebut disajikan pada Tabel 4.9 

UNIVERSITAS PAMULANG 

67 

Tabel 4.9 30- _run_ Optimasi GWO 

|Run|n<br>(wolf)|Iteration|MAPE (%)|w1<br>(MA)|w2 (ES)|w3<br>(RNN)|
|---|---|---|---|---|---|---|
|1|20|100|40.092401%|0.315733|0.000000|0.684267|
|2|20|100|40.092405%|0.315699|0.000000|0.684301|
|3|20|100|40.092417%|0.315730|0.000007|0.684263|
|4|20|100|40.092401%|0.315733|0.000000|0.684267|
|5|20|100|40.092403%|0.315762|0.000000|0.684238|
|6|20|100|40.092401%|0.315737|0.000000|0.684263|
|7|20|100|40.092401%|0.315739|0.000000|0.684261|
|8|20|100|40.092405%|0.315747|0.000001|0.684251|
|9|20|100|40.092403%|0.315761|0.000000|0.684239|
|10|20|100|40.092403%|0.315758|0.000000|0.684242|
|11|20|100|40.092417%|0.315771|0.000006|0.684224|
|12|20|100|40.092402%|0.315748|0.000000|0.684252|
|13|20|100|40.092401%|0.315734|0.000000|0.684266|
|14|20|100|40.092407%|0.315746|0.000002|0.684252|
|15|20|100|40.092404%|0.315710|0.000000|0.684290|
|16|20|100|40.092405%|0.315700|0.000000|0.684300|
|17|20|100|40.092403%|0.315764|0.000000|0.684236|
|18|20|100|40.092401%|0.315727|0.000000|0.684273|
|19|20|100|40.092401%|0.315730|0.000000|0.684270|
|20|20|100|40.092418%|0.315713|0.000006|0.684280|
|21|20|100|40.092402%|0.315720|0.000000|0.684280|



UNIVERSITAS PAMULANG 

|22|20|100|40.092403%|0.315715|0.000000|0.684285|
|---|---|---|---|---|---|---|
|23|20|100|40.092401%|0.315733|0.000000|0.684267|
|24|20|100|40.092402%|0.315723|0.000000|0.684277|
|25|20|100|40.092402%|0.315722|0.000000|0.684278|
|26|20|100|40.092406%|0.315692|0.000000|0.684308|
|27|20|100|40.092401%|0.315732|0.000000|0.684268|
|28|20|100|40.092441%|0.315719|0.000016|0.684265|
|29|20|100|40.092402%|0.315746|0.000000|0.684254|
|30|20|100|40.092406%|0.315752|0.000002|0.684247|



Supaya dapat lebih mudah dalam menganalisa hasil optimasi ini, dibuatkan ringkasan statistik terhadap hasil pengujian 30 run. 

Tabel 4.10. Ringkasan Statistik hasil Optimasi GWO 

|**Parameter**<br>**Statistik**|**Nilai Fitness**<br>**(MAPE)**|**Bobot w1**<br>**(MA)**|**Bobot w2**<br>**(ES)**|**Bobot w3**<br>**(RNN)**|
|---|---|---|---|---|
|**Terbaik**|40,092401%|0,315732|0,000000|0,684268|
|**Terburuk**|40,092441%|0.315719|0.000016|0.684265|
|**Rata-rata**|40,092405%|0,315733|0,000001|0,684265|
|**Std. Deviation**|0,000008%|0,000020|0,000003|0,000020|



Berdasarkan ringkasan statistik hasil optimasi GWO di Tabel 4.10, Algoritma GWO mencapai nilai fitness (MAPE) terbaik sebesar 40,092401%, MAPE terburuk sebesar 40,092441%, dan rata-rata dari nilai MAPE yang dihasilkan dari seluruh run adalah sebesar 40,092405%. Standar Deviasi MAPE berada pada angka yang sangat kecil, yaitu 0,000008%. Nilai deviasi yang mendekati nol ini mempunyai arti bahwa algoritma GWO cukup stabil karena 

UNIVERSITAS PAMULANG 

69 



<!-- Start of picture text -->
Tabel Konvergensi GWO<br>40.45 MAPE: 40.092401%<br>40.40<br>40.35<br>® 40.30<br>Ww<br>a<br><<br>=<br>4 40.25<br>vu<br>oO<br>40.20<br>40.15<br>40.10<br>0 20 40 60 80 100<br>Iteration<br><!-- End of picture text -->

~~<u>a</u>~~ 

model _baseline_ . Selain itu juga dibahas seberapa besar model _ensemble_ dapat memperbaiki performa dari model _baseline_ yang terbaik. 

Dominasi bobot RNN yang sebesar 68,43%, MA sebesar 31,57%, dan ES sebesar 0,00%. Kombinasi bobot ini menunjukkan bahwa GWO memanfaatkan kekuatan RNN dalam menangkap pola non-linear dan memberikan porsi berarti pada MA untuk menangkap komponen linear dan tren data secara efektif, meskipun secara individu MA memiliki MAPE lebih tinggi (44,46%). ES (MAPE 46,41%) memiliki performa paling rendah di antara ketiga model baseline, sehingga GWO memberikan bobot 0,00% karena tidak memberikan kontribusi positif terhadap peningkatan akurasi ensemble. RNN (MAPE 41,35%) memiliki kemampuan menangkap pola non-linear melalui arsitekturnya, sehingga menjadi model individu terbaik. GWO memberikan bobot 68,43% karena performanya yang superior dalam menangkap pola data. Proses optimasi yang menghasilkan konfigurasi bobot tersebut divisualisasikan pada kurva konvergensi GWO di Gambar 4.6. 

##### **4.2.1 Bobot Ensemble** 

Model Recurrent Neural Network (RNN) mendapatkan bobot paling dominan, yakni mencakup sekitar 68,43% dari total bobot ensemble. Dominasi bobot pada RNN mengindikasikan bahwa algoritma GWO mengenali RNN sebagai model prediktif terbaik dalam menangkap pola data yang diteliti. Sementara itu, model Moving Average (MA) turut memberikan kontribusi sebesar 31,57%, sedangkan Exponential Smoothing (ES) tidak dipergunakan dengan bobot 0,00%. 

##### **4.2.2 Perbandingan Performa Model** 

Analisis performa difokuskan pada perbandingan langsung antara model ensemble teroptimasi GWO dengan ketiga model individu pendukungnya, yaitu Moving Average (MA), Exponential Smoothing (ES), dan arsitektur Recurrent Neural Network (RNN). Tujuan dari komparasi ini adalah untuk mengukur sejauh mana algoritma GWO mampu mereduksi tingkat kesalahan prediksi dibandingkan model individual. Rangkuman metrik evaluasi dari seluruh model disajikan pada Tabel 4.11 Perbandingan visual dari seluruh 

UNIVERSITAS PAMULANG 

71 



<!-- Start of picture text -->
Grafik perbandingan MAPE semua model<br>46.41%<br>44.46%<br>41.35%<br>40.09%<br>40<br>30<br>g<br>u<br>L<br>FsE20<br>10<br>0<br>MA és RNN GWO Ensemble<br>Model<br><!-- End of picture text -->

~~<u>a</u>~~ 



<!-- Start of picture text -->
Prediksi Ensemble GWO vs Data Uji<br>|<br>0.8<br>7 Wi<br><!-- End of picture text -->

model MA (31,57%) dan RNN (68,43%) untuk menghasilkan prediksi yang lebih akurat. 

##### **4.2.3 Peningkatan Akurasi Model GWO Ensemble** 

Peningkatan akurasi yang dicapai oleh GWO Ensemble menunjukkan bahwa algoritma GWO mampu mengkombinasikan kelebihan MA (31,57%) dan RNN (68,43%) untuk menghasilkan model ensemble yang lebih unggul secara keseluruhan dibandingkan model individu terbaik (RNN). GWO membuktikan bahwa meskipun RNN sudah memiliki performa baik, dengan mengkombinasikannya dengan MA, dapat memberikan hasil yang lebih akurat. 

UNIVERSITAS PAMULANG 

74 

#### **BAB V** 

#### **KESIMPULAN DAN SARAN** 

##### **5.1 Kesimpulan** 

Penelitian ini telah berhasil melakukan optimasi bobot pada model _ensemble_ untuk peramalan penjualan dengan menggunakan _Grey Wolf Optimizer_ (GWO).  Kesimpulan yang dapat diambil dari penelitian ini adalah sebagai berikut: 

1. Penelitian ini telah berhasil mengembangkan model weighted ensemble yang menggabungkan tiga model baseline — Moving Average (MA), Exponential Smoothing (ES), dan Recurrent Neural Network (RNN). Hasil optimasi menunjukkan bahwa RNN memberikan kontribusi bobot paling dominan sebesar 68,43%, MA sebesar 31,57%, sedangkan ES memperoleh bobot 0,00%. Analisis terhadap masing-masing bobot adalah sebagai berikut: RNN mendapat bobot tertinggi karena memiliki MAPE terendah (41,35%) di antara ketiga model baseline dan kemampuan superior dalam menangkap pola non-linear dan sekuensial data penjualan melalui arsitektur hidden statenya, sehingga GWO menentukan RNN sebagai kontributor utama dalam meminimalkan error prediksi. MA memperoleh bobot 31,57% karena MA efektif dalam menangkap komponen linear dan tren jangka pendek sehingga memberikan informasi komplementer yang tidak sepenuhnya tertangkap oleh RNN. ES memperoleh bobot 0,00% karena memiliki MAPE tertinggi (46,41%) dan tidak memberikan kontribusi positif pada MAPE _ensemble_ . 

2. Algoritma Grey Wolf Optimizer (GWO) berhasil menemukan kombinasi bobot optimal dengan meminimalkan kesalahan prediksi. Model ensemble teroptimasi menghasilkan MAPE sebesar 40,09%, melampaui kinerja seluruh model baseline individu. Dibandingkan best baseline RNN (MAPE 41,35%), terjadi peningkatan akurasi sebesar 1,26%. 

3. Model GWO Ensemble unggul pada metrik MAPE (40,09% vs RNN 41,35%), namun pada metrik lainnya GWO Ensemble berada di posisi kedua setelah ES: MAE 0,1184 (terbaik ES 0,1074), MSE 0,0262 (terbaik ES 

UNIVERSITAS PAMULANG 

75 

0,0183), RMSE 0,1620 (terbaik ES 0,1352), dan R² -0,3241 (terbaik ES 0,0783). Hal ini mengindikasikan bahwa GWO Ensemble memprioritaskan penurunan MAPE sebagai fungsi objektif, sementara ES tetap unggul dalam meminimalkan error absolut dan varian residual. 

4. Hasil pengujian multi-run sebanyak 30 kali membuktikan bahwa algoritma GWO sangat robust dengan standar deviasi MAPE yang sangat kecil, yaitu 0,000008%. Algoritma GWO tidak rentan terhadap variasi inisialisasi awal acak, mampu menghindari jebakan optimum lokal, dan konsisten dalam mencapai konvergensi global pada setiap percobaan. 

##### **5.2 Saran** 

Berdasarkan hasil penelitian dan pengalaman selama proses analisis,  terdapat keterbatasan pada penelitian ini, ada beberapa saran yang perlu diperhatikan untuk penelitian selanjutnya, baik oleh peneliti berikutnya maupun oleh pihak yang mengkonsumsi hasil peramalan penjualan, yaitu : 

1. Sebagai saran akademis, mengingat model berbasis jaringan saraf (RNN) memiliki kelebihan dalam menangkap pola non-linear data penjualan, penelitian selanjutnya disarankan untuk menggunakan arsitektur _deep learning_ yang lebih mutakhir sebagai model _baseline_ dalam _ensemble_ , seperti _Long Short-Term Memory_ (LSTM) yang mampu mengatasi masalah _vanishing gradient_ melalui mekanisme _gate_ sehingga dapat menangkap ketergantungan jangka panjang ( _long-term dependencies_ ) pada data penjualan, _Gated Recurrent Unit_ (GRU) sebagai varian yang lebih ringan secara komputasi dengan jumlah parameter lebih sedikit, atau _Transformer_ yang memanfaatkan mekanisme _self-attention_ untuk menangkap pola ketergantungan antar waktu secara lebih fleksibel tanpa keterbatasan urutan sekuensial. 

2. Sebagai saran praktis, penelitian selanjutnya dapat mengaplikasikan data _multivariate_ dengan menyertakan variabel-variabel tambahan seperti harga produk, promo, hari libur, dan data musiman yang dapat memengaruhi pola penjualan. Pendekatan ini memungkinkan model menangkap hubungan 

UNIVERSITAS PAMULANG 

76 



<!-- Start of picture text -->
Pendekatan UNIVARIATE Pendekatan MULTIVARIATE<br>HANYA SATU VARIABEL: DATA PENJUALAN BANYAK VARIABEL: DATA KAYA & KONTEKSTUAL<br>oy TIDAK MENANGKAP FAKTOR LAIN </_ MENANGKAP BANYAK FAKTOR i<br>ix ABAIKAN PROMO, HARGA, HARI LIBUR Wecan v PENGARUHPROMOHariLibur& LIBUR TERPETAKANKategori f-<br>pe —) aL ><br>yi @ ill PREDIKSI|alll AGREGAT(MA/ES/RNN)(TIDAK GRANULAR) E WiesPREDIKSI (Time SPESIFIKbctlee & AKURATVatucal Etat)<br>mm, PREDIKSI TERBATAS f°)ES wh PREDIKSI GRANULAR & AKURAT<br>f lis (MAPE 2 40%) oH | (Per Produk, Toko, & Periode)<br>« SEBELUM OPTIMASI » SESUDAH OPTIMASI<br>INFOGRAFIS: TRANSFORMASI PERAMALAN PENJUALAN - DARI UNIVARIATE KE MULTIVARIATE<br><!-- End of picture text -->

~~EEE _________________s~~ 

#### **DAFTAR PUSTAKA** 

- Adhikari, R., & Agrawal, R. K. (2012). A novel weighted ensemble technique for time series forecasting. _Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)_ , _7301 LNAI_ (PART 1), 38–49. https://doi.org/10.1007/9783-642-30217-6_4 

- Afifudin, M., Junaidi, A., Sihananto, A. N., & Fithriyah, I. (2024). Gwo-Svm: an Approach To Improving Svm Performance Using Grey Wolf Optimizer in Intellectual Disability Classification. _Jurnal Informatika Dan Teknik Elektro Terapan_ , _12_ (3S1), 4440–4453. https://doi.org/10.23960/jitet.v12i3s1.5359 

- Apriandi, N. D., Soleh, A., & Irwanto, T. (2023). _The Effect Of Application Of Aida (Attention, Interest, Desire AndAction) On Telkomsel Card Purchase Decisions In Bengkulu CityPengaruh Penerapan Aida (Attention, Interest, Desire Dan Action)Terhadap Keputusan Pembelian Kartu Telkomsel Di Kota Bengkulu_ . _2_ (2), 189–202. https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&u rl=https://jurnal.unived.ac.id/index.php/jambd/article/download/4379/342 3/&ved=2ahUKEwinhujA7_2NAxUHRWcHHUIAFa8QFnoECCMQAQ &usg=AOvVaw3W8mlR39P4hDZW_1rao6-E 

- Aras, S., Deveci Kocakoç, İ., & Polat, C. (2017). Comparative study on retail sales forecasting between single and combination methods. _Journal of Business Economics and Management_ , _18_ (5), 803–832. https://doi.org/10.3846/16111699.2017.1367324 

- Armstrong, J. S. (2001a). Combining forecasts. _Economic Forecasting_ , _January 2001_ , 85–108. https://doi.org/10.1017/cbo9780511628603.004 

- Armstrong, J. S. (2001b). _Principles of Forecasting: A Handbook for researchers and pratitioners_ . 

- Azizah, N. N., & Nisah, F. A. (2024). Analisis Peramalan Demand Produk RBL dengan Metode Double Exponensial Smoothing, Moving Avarage, dan Regresi Linear di PT Seiwa Indonesia. _Jurnal Teknik Industri Terintegrasi_ , _7_ (1), 215–324. https://doi.org/10.31004/jutin.v7i1.24763 

- Dietterich, T. G. (2000). _Ensemble Method in Machine Learning_ . 

- Eberhart, R., & Kennedy, J. (1995). A new optimizer using particle swarm theory. _MHS’95. Proceedings of the Sixth International Symposium on Micro Machine and Human Science_ , 39–43. https://doi.org/10.1109/MHS.1995.494215 

- EL Mahjouby, M., El Fahssi, K., Taj Bennani, M., Lamrini, M., & El Far, M. (2024). Simple RNN-LSTM hybrid deep learning model for Bitcoin and EUR_USD forecasting. _TELKOMNIKA (Telecommunication Computing Electronics and Control)_ , _23_ (1), 175. https://doi.org/10.12928/telkomnika.v23i1.25925 

UNIVERSITAS PAMULANG 

78 

Ganguly, P., & Mukherjee, I. (2024). Enhancing Retail Sales Forecasting with Optimized Machine Learning Models. _4th International Conference on Sustainable Expert Systems, ICSES 2024 - Proceedings_ , 884–889. https://doi.org/10.1109/ICSES63445.2024.10762950 

George E.P. Box, G. M. J. (2014). _Time series analysis: Forecasting and control_ . 

Hamouda, E., & Tarek, M. (2024). A hybrid approach of ensemble learning and grey wolf optimizer for DNA splice junction prediction. _PLoS ONE_ , _19_ (9 September), 1–18. https://doi.org/10.1371/journal.pone.0310698 

- Heizer, J., Render, B., & Munson, C. (2014). _Operations Management Sustainibility and Supply Chain Management_ . 

Husayn, M., Adegboye, O. R., & Alzubi, A. (2025). GWO-Optimized Ensemble Learning for Interpretable and Accurate Prediction of Student Academic Performance in Smart Learning Environments. _Applied Sciences (Switzerland)_ , _15_ (22). https://doi.org/10.3390/app152212163 

- Lou, L., Xia, W., Sun, Z., Quan, S., Yin, S., Gao, Z., & Lin, C. (2023). COVID19 mortality prediction using ensemble learning and grey wolf optimization. _PeerJ Computer Science_ , _9_ (2020), 1–18. https://doi.org/10.7717/PEERJ-CS.1209 

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. _International Journal of Forecasting_ , _36_ (1), 54–74. https://doi.org/10.1016/j.ijforecast.2019.04.014 

Manokaran, J., & Vairavel, G. (2023). IGWO-SoE: Improved Grey Wolf Optimization Based Stack of Ensemble Learning Algorithm for Anomaly Detection in Internet of Things Edge Computing. _IEEE Access_ , _11_ (September 2023), 106934–106953. https://doi.org/10.1109/ACCESS.2023.3319814 

- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer. _Advances in Engineering Software_ , _69_ , 46–61. https://doi.org/10.1016/j.advengsoft.2013.12.007 

- Muro, C., Escobedo, R., Spector, L., & Coppinger, R. P. (2011). Wolf-pack (Canis lupus) hunting strategies emerge from simple rules in computational simulations. _Behavioural Processes_ , _88_ (3), 192–197. https://doi.org/https://doi.org/10.1016/j.beproc.2011.09.006 

- Narasimhan, G., & Victor, A. (2024). Grey wolf optimized stacked ensemble machine learning based model for enhanced efficiency and reliability of predicting early heart disease. _Automatika_ , _65_ (3), 749–762. https://doi.org/10.1080/00051144.2024.2317098 

- Rob J. Hyndman, & Athanasopoulos, G. (2018). _Forecasting: Principles and practice_ (2nd ed.). 

Saheed, Y. K., & Misra, S. (2024). A voting gray wolf optimizer-based ensemble learning models for intrusion detection in the Internet of Things. 

UNIVERSITAS PAMULANG 

79 

_International Journal of Information Security_ , _23_ (3), 1557–1581. https://doi.org/10.1007/s10207-023-00803-x 

- SVSV Prasad Sanaboina, M Chandra Naik, K Rajiv. (2025). An Advanced Ensemble Framework Employing Grey Wolf Optimization and Feature Selection Techniques for Enhanced Intrusion Detection on Unbalanced NSL-KDD Data. _Communications on Applied Nonlinear Analysis_ , _32_ (3), 865–878. https://doi.org/10.52783/cana.v32.4627 

- ZhiQiang Zeng, Saratha Sathasivam, J. X. & H. Z., & We. (2026). _Scientific Reports Article in Press Real-time dynamic prediction of HFMD transmission using SEIRQ-ARIMA hybrid model optimized by multi-stage ABC-GWO algorithm IN AR IN_ . 

UNIVERSITAS PAMULANG 

80 

#### **DAFTAR RIWAYAT HIDUP** 

|1. Nama|: Bayu Nurcahyono|NIM|: 231012000006|
|---|---|---|---|
|2. Tempat/tgl.<br>Lahir|: Surakarta, 13 Juli 1979||Pria<br>Wanita|
|3. Agama|: Kristen|||
|Jenjang<br>Prodi/Jurusan<br>Perguruan Tinggi<br>Tahun Lulus|:S1<br>:Teknik Informatika<br>:ITB<br>:2002|||
|5. Alamat rumah|:Villa GunungLestari||No HP|
||Jl. Rinjani II Blok F2/5|15414||
||Jombang, Ciputat||e-mail|
||Tangerang Selatan 15414|mirmath|eorara@gmail.com|
|6. Web pribadi|:<br>https://www.linkedin.com/in/<br>bayu-nurcahyono-<br>bab462210/||No. Telp|
|7. Pekerjaan|:Product Head|||
|8. Alamat Kantor|: Plaza Mutiara, Lt 8||e-mail|
||Kuningan, Jakarta Selatan<br>12950|bayu.nur<br>co.id|cahyono@technohub.|
|9. Web kantor|:www.technohub.co.id|||



Demikian daftar riwayat hidup ini dibuat dengan sebenarnya. 

Tangerang Selatan, 18 Juli 2026 

(Bayu Nurcahyono) 

UNIVERSITAS PAMULANG 

81 

