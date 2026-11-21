# Keputusan Design dari CART

## Alasan?
> Salah satu "Issue in DTL" adalah terkait denominator pada GainRatio(SplitInformation) pada C4.5, sehingga terbentuk keharusan untuk membandingkan si InformationGain dengan rata-rata.

## Bagaimana dengan CART? 
> CART tidak memiliki penalti denominator dan CART cukup menghitung nilai kuadrat (Gini), tidak perlu memakai fungsi log. Selain itu, CART sendiri adalah pohon biner sehingga perlakuan  dari pohon ini lebih stabil dibanding C4.5 yang bisa memiliki beberapa anak.

## Optimasi CART
> Pada kode ini, optimasi dilakukan dengan melakukan Cost Complexity Pruning dengan Cross Validation. Penjelasan algoritma bisa di cek di 13.2.7 dan 13.2.8. Selain itu, kami tidak menerapkan Weighted Gini Impurity, karena pada dasarnya (secara baku), praktik terbaik dari scikit sendiri adalah menggunakan param class_weight.

## Penanganan Kategorikal dan Null Values
> Hal unik yang saya temukan ketika melakukan *surfing* (intinya *search*) di Google, kebanyakan referensi langsung mengasumsikan dengan pemakaian threshold. Ini karena pada dasarnya CART juga dikonsider sebagai model pohon regresi, sehingga fitur kategorikal tidak begitu dihighlight. Setelah berdiskusi dengan teman, saya mendapatkan buku klasik dari Breiman, yang menjelaskan mengenai kedua hal ini. Pertama, terkait masalah kategorikal, secara intuitif (kalau manusia yang mengerjakan), kita bisa mengklasifikasikannya sendiri dengan membaca. Lalu, bagaimana dengan mesin atau komputer--semiripnya? Approach awal yang saya pikirkan adalah dengan encoding, tapi sepertinya ini tidak efektif, dan memang disarankan pada buku tersebut untuk tetap mempertahankan *object type*-nya sebagai object/string. Sementara itu, penanganan missing value memanfaatkan konsep *surrogate split*, yang mencari fitur pengganti yang bisa ikut merepresentasi pembagian kelas berdasarkan missing value tersebut.

## Referensi Utama:
- Buku: Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and regression trees. Monterey, CA: Wadsworth and Brooks. 
- Analyzing CART: https://arxiv.org/abs/1906.10086
- CART (Classification And Regression Trees): https://github.com/Valdecy/CART
- Kool Beans: https://www.causalmlbook.com/classification-and-regression-trees-cart.html#additional-considerations-and-extensions-for-regression-trees
- Permasalahan class imbalance pada sebuah node: https://imbalanced-learn.org/stable/auto_examples/applications/plot_impact_imbalanced_classes.html

## Penggunaan AI
> Pada bagian source code untuk CART, AI hanya dimanfaatkan dalam pencarian sumber.
