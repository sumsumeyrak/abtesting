"""İş problemi
Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme
türüne alternatif olarak yeni bir teklif türü olan "averagebidding"’i tanıttı.
Müşterilerimizden biri olanbombabomba.com, bu yeni özelliği test etmeye karar verdi
ve averagebidding inmaximumbidding'den daha fazla dönüşüm getirip getirmediğini
anlamak için bir A/B testiyapmak istiyor.
A/B testi 1 aydır devam ediyor
ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.
Bombabomba.com için nihai başarı ölçütü Purchase dır.
Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.


Veri Seti Hikayesi

Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri
ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra
buradan gelen kazanç bilgileri yer almaktadır.
Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır.
Bu veri setleri ab_testing.xlsx excel’ininayrı sayfalarında yer almaktadır.
Kontrol grubuna Maximum Bidding, test grubuna AverageBidding uygulanmıştır.


mpressionReklam: Görüntüleme sayısı
Click          : Görüntülenen reklama tıklama sayısı
Purchase       : Tıklanan reklamlar sonrası satın alınan ürün sayısı
Earning        : Satın alınan ürünler sonrası elde edilen kazanç
"""
#Görev 1:  Veriyi Hazırlama ve Analiz Etme

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,\
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None




#Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri
#setini okutunuz.
# Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

control_df = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Control Group")
test_df = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Test Group")

#Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
control_df.describe().T
test_df.describe().T

# ==========================
# Exploratory Data Analysis
# ==========================

def check_df(dataframe, head=5, boxplt=False, column="Purchase"):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Types #####################")
    print(dataframe.info())
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### MissingValues #######################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)
    if boxplt == True:
        sns.boxplot(df[column])
        print(plt.show())
check_df(control_df,boxplt=True)

check_df(test_df,boxplt=True)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car



#sms.DescrStatsW(control_df["Purchase"]).tconfint_mean()
#sms.DescrStatsW(test_df["Purchase"]).tconfint_mean()

#Adım 3: Analiz işleminden sonra concatmetodunu kullanarak kontrol ve
#test grubu verilerini birleştiriniz.
control_df.columns = [col + "-Cont" for col in control_df.columns]
control_df.head()
test_df.columns = [col + "-Test" for col in test_df.columns]
test_df.head()
df=pd.concat([control_df,test_df],axis=1)
df.head()

grab_col_names(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)



#Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2
# H1 : M1!= M2

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

#Adım 2: Kontrol ve test grubu için purchase(kazanç)
# ortalamalarını analiz ediniz.
test_stat, pvalue = shapiro(df["Purchase-Test"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df["Purchase-Cont"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.
#p value 0.05 ten büyük olduğu için H0 hipotezi kabul edilmiştir.Yani normallik varsayımı
#sağlanmıştır.
# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

# Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
test_stat, pvalue = ttest_ind(df["Purchase-Test"],
                              df["Purchase-Cont"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p value 0.05'ten büyük.Böylece H0 hipotezi kabul edilir.