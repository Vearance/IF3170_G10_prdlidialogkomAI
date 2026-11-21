# Keputusan Design dari CART

## Alasan?
> Salah satu "Issue in DTL" adalah terkait denominator pada GainRatio(SplitInformation) pada C4.5, sehingga terbentuk keharusan untuk membandingkan si InformationGain dengan rata-rata.

## Bagaimana dengan CART? 
> CART tidak memiliki penalti denominator dan CART cukup menghitung nilai kuadrat (Gini), tidak perlu memakai fungsi log. Selain itu, CART sendiri adalah pohon biner sehingga perlakuan  dari pohon ini lebih stabil dibanding C4.5 yang bisa memiliki beberapa anak.

## Struktur Algoritma 

## Optimasi CART
> Pada kode ini, optimasi dilakukan dengan melakukan Cost Complexity Pruning dengan Cross Validation. Penjelasan algoritma bisa di cek di 13.2.7 dan 13.2.8.

## Referensi Utama:
- Analyzing CART: https://arxiv.org/abs/1906.10086
- CART (Classification And Regression Trees): https://github.com/Valdecy/CART
- Kool Beans: https://www.causalmlbook.com/classification-and-regression-trees-cart.html#additional-considerations-and-extensions-for-regression-trees
- Permasalahan class imbalance pada sebuah node: https://imbalanced-learn.org/stable/auto_examples/applications/plot_impact_imbalanced_classes.html

## Penggunaan AI
> Pada bagian source code untuk CART, AI hanya dimanfaatkan dalam pencarian sumber.