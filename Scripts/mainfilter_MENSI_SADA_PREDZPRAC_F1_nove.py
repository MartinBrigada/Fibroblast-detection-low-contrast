#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      martin
#
# Created:     13.03.2024
# Copyright:   (c) martin 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#vychazi se z DATALIST souboru

import numpy as np
import math
import cv2
import os
import copy
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.ndimage import convolve
from collections import defaultdict

def Ziskej_obr(cesta,typ):
    seznam_obrazku = [os.path.join(cesta, f) for f in os.listdir(cesta) if f.endswith('.' + typ)] 
    return seznam_obrazku

def Zobraz(img):
    fig = plt.figure()
    plt.imshow(img,cmap='gray')
    plt.axis('off')  
    plt.show()
    
def Zobraz_RGB(img):
  # Zkontroluj, zda je obrázek 16bitový
    if img.dtype == np.uint16:
        # Převeď 16bitové hodnoty na 8bitové (0-255)
        img = (img / 256).astype(np.uint8)  # Dělením na 256 převedeme hodnoty na rozsah 0-255

    # Zobraz obrázek
    fig = plt.figure()
    plt.imshow(img)  # Automaticky rozpozná, zda je obrázek barevný nebo černobílý
    plt.axis('off')  
    plt.show()

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def Nacti_souradnice(csv_soubor,cislo_obr):
    soubor = os.path.join(os.getcwd(),csv_soubor)
    # df = pd.read_csv(soubor)

    with open(soubor,'r') as f:
        radky = f.readlines()

    data_list = [] #elipsa od data_list[X][20]
    for line in radky:
            # Rozdělení řádku podle čárek
            items = line.strip().split(',')
            if items[8] == str(cislo_obr):
                data_list.append(items)    
    return data_list

def Nakresli_znacky(image,data_list):
    # cv2.imshow('Filtrovany obrazek bez znacek', image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for pt in data_list:
                a, b, r = round(float(pt[4])), round(float(pt[5])), round(float(pt[9]))
          
                # Draw the circumference of the circle.
                # cv2.circle(obr, (a, b), radius=r, color=(0, 255, 0), thickness=5)
          
                # Draw a small circle (of radius 1) to show the center.
                # cv2.circle(obr, (a, b), radius=1, color=(0, 0, 255), thickness=3)
                cv2.circle(image, (a, b), radius=1, color=(0, 0, 255), thickness=1) #tecky
                # center_coor = (round(float(pt[20])),round(float(pt[21])))
                center_coor = (a,b)
                axe_len =(round(float(pt[22])),round(float(pt[23])))
                angle = math.degrees(float(pt[24]))
                startAngle = 0
                endAngle = 360
                cv2.ellipse(image,center_coor,axe_len,angle,startAngle,endAngle,color=(255,255,0),thickness=1)
    # cv2.imshow('Filtrovany obrazek + znacky', image)

def Nakresli_znacky_stred(image,data_list):
    # cv2.imshow('Filtrovany obrazek bez znacek', image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for pt in data_list:
                a, b, r = round(float(pt[4])), round(float(pt[5])), round(float(pt[9]))
                cv2.circle(image, (a, b), radius=1, color=(0, 255, 255), thickness=2) #zluta, tecky stredove

def equalize_histogram_16bit(image):
    # Definice počtu intervalů
    num_bins = 65536  # Celý rozsah 16 bitů

    # Výpočet histogramu pro 16bitový obrázek
    hist, _ = np.histogram(image.flatten(), bins=num_bins, range=[0, 65535])

    # Kumuulativní distribuční funkce
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]

    # Vytvoření mapovací tabulky
    equalize_map = np.interp(image.flatten(), np.arange(0, 65536), cdf_normalized * 65535)
    equalize_map = np.uint16(equalize_map.reshape(image.shape))

    # Aplikace mapovací tabulky pro equalizaci histogramu
    equalized_image = equalize_map

    return equalized_image

def wiener_filter(image, kernel_size, noise_variance):    
    # Převedení obrazu a jádra na float32
    image = np.float32(image)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)

    # Výpočet Fourierových transformací
    f_image = np.fft.fft2(image)
    f_kernel = np.fft.fft2(kernel,s=image.shape)

    # Výpočet mocnin Fourierových transformací
    f_image_squared = np.abs(f_image) ** 2
    f_kernel_squared = np.abs(f_kernel) ** 2

    # Wienerův filtr
    H = np.conj(f_kernel) / (f_kernel_squared + noise_variance)
    f_deconvolved = H * f_image
    deconvolved = np.fft.ifft2(f_deconvolved).real

    # Normalizace do rozsahu 0-255 ---- zatim do 65536
    deconvolved = np.uint8(np.clip(deconvolved, 0, 255))

    return deconvolved

def predzprac_ica_OK(obr):
    emc2_image_cv2 = obr
    ica_cv2 = FastICA(n_components = 1,whiten='arbitrary-variance')
    # reconstruct image with independent components
    emc2_image_ica_cv2 = ica_cv2.fit_transform(emc2_image_cv2)  #komponenty
    emc2_restored_cv2 = ica_cv2.inverse_transform(emc2_image_ica_cv2)
    normalized_img_cv2 = normalize_image(emc2_image_cv2-emc2_restored_cv2)
    normalized_img_cv2 = normalized_img_cv2 - normalized_img_cv2.min()
    normalized_img_cv2 = normalized_img_cv2 / normalized_img_cv2.max() * 65536
    new_img = np.uint16(normalized_img_cv2)
    return new_img

# Funkce pro výpočet Haarových funkcí
def haar_features(image, window_size):
    # Definování Haarových funkcí (vertikální, horizontální a diagonální)
    haar_vertical = np.array([[1], [-1]])  # Jednoduchý vertikální filtr
    haar_horizontal = np.array([[1, -1]])  # Jednoduchý horizontální filtr
    haar_diagonal = np.array([[1, -1], [-1, 1]])  # Diagonální filtr

    # Aplikace filtrů (konvoluce)
    response_vertical = convolve(image, haar_vertical)
    response_horizontal = convolve(image, haar_horizontal)
    response_diagonal = convolve(image, haar_diagonal)

    # Výpočet maximální odpovědi
    features = np.maximum(np.maximum(np.abs(response_vertical), np.abs(response_horizontal)), np.abs(response_diagonal))
    return features

# Funkce pro tvorbu PP obrazu
def compute_pp_image(image, window_size):
    # Normalizace obrazu na rozsah [0, 1]
    normalized_image = image.astype(np.float32) / 255.0

    # Výpočet Haarových funkcí
    haar_response = haar_features(normalized_image, window_size)

    # Prahování a vytvoření pravděpodobnostního obrazu
    # Použití slabého prahu (např. 0.5) pro rozlišení částic od pozadí
    particle_prob_map = np.where(haar_response > 0.05, 1, 0)

    return particle_prob_map


def vyhodnot_kontury_puvodni(contours, data_list, tolerance=-5):
    """
    Vyhodnotí přesnost detekce kontur na základě středových souřadnic.

    Parameters:
    - contours: Seznam kontur.
    - data_list: Seznam středových souřadnic buněk (každá položka má hodnoty na indexech 4 a 5).

    Returns:
    - metrics: Slovník s hodnotami Precision, Recall a F1 skóre.
    """
    
    # Převod 16bitového obrázku na 8bitový pro vykreslování
    image_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

    # Převod na RGB, aby bylo možné vykreslovat barevné kontury
    image_with_contours = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    # Převést středové souřadnice na seznam tuple (px, py)
    centers = [(round(float(data[4])), round(float(data[5]))) for data in data_list]
    
    # Vytvořit seznam pro označení středových bodů, které byly použity
    detected_centers = [False] * len(centers)

    # Projít každou konturu
    for contour in contours:
        found_center = False  # Zda byla v této kontuře nalezena nějaká středová souřadnice

        for i, (px, py) in enumerate(centers):
            if not detected_centers[i]:  # Pokud tento bod nebyl zatím přiřazen
                # Pokud bod leží uvnitř kontury (nebo mírně mimo ni)
                if cv2.pointPolygonTest(contour, (px, py), True) >= tolerance:  # Tolerance 5 pixelů
                    tp += 1
                    detected_centers[i] = True
                    found_center = True

        # Pokud kontura nemá žádný středový bod, je to false positive
        if not found_center:
            fp += 1

    # Zbývající středové body, které nebyly přiřazeny, jsou false negatives
    fn = detected_centers.count(False)

    # Výpočet metrik
    # precision = tp / (tp + fp) if tp + fp > 0 else 0
    # recall = tp / (tp + fn) if tp + fn > 0 else 0
    # f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        # "Precision": precision,
        # "Recall": recall,
        # "F1 Score": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }


def vyhodnot_kontury(contours, data_list,image, tolerance=5):
    """
    Vyhodnotí přesnost detekce kontur na základě středových souřadnic.

    Parameters:
    - contours: Seznam kontur.
    - data_list: Seznam středových souřadnic buněk (každá položka má hodnoty na indexech 4 a 5).

    Returns:
    - metrics: Slovník s hodnotami Precision, Recall a F1 skóre.
    """
    
    # Převod 16bitového obrázku na 8bitový pro vykreslování
    image_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

    # Převod na RGB, aby bylo možné vykreslovat barevné kontury
    image_with_contours = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    # Převést středové souřadnice na seznam tuple (px, py)
    centers = [(round(float(data[4])), round(float(data[5]))) for data in data_list]
    
    # Vytvořit seznam pro označení středových bodů, které byly použity
    detected_centers = [False] * len(centers)

    # Projít každou konturu
    for contour in contours:
        found_center = False  # Zda byla v této kontuře nalezena nějaká středová souřadnice

        for i, (px, py) in enumerate(centers):
            if not detected_centers[i]:  # Pokud tento bod nebyl zatím přiřazen
                # Pokud bod leží uvnitř kontury (nebo mírně mimo ni)
                pom = cv2.pointPolygonTest(contour, (px, py), True)
                if (pom >= 0 or (pom < 0 and (pom >= -tolerance))): # bud je to uvnitr, nebo ve vzdalenosti pod toleranci 5 
                    tp += 1
                    cv2.drawContours(image_with_contours, [contour], -1, (0,255,0), 1) #TP zelena
                    detected_centers[i] = True
                    found_center = True
        # Pokud kontura nemá žádný středový bod, je to false positive
        if not found_center:
            fp += 1
            cv2.drawContours(image_with_contours, [contour], -1, (0,0,255), 1) #FP cervena
        

    # Zbývající středové body, které nebyly přiřazeny, jsou false negatives
    fn = detected_centers.count(False)
    

    # Výpočet metrik
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }, image_with_contours

def Zkontroluj_slozku(cesta):
    if not os.path.isdir(cesta):
        os.makedirs(cesta)

#*********************************************************************************************************
poradi = [0,1] # jedna nebo druha sada
slozka = ['B1_04_1_2_Bright_field','B1_04_1_4_Bright_field']
slozka_ulozeni = ['B1_04_1_2_F1_SCORE_KONTURY','B1_04_1_4_F1_SCORE_KONTURY']
csv_soubor = ['B1_04_2_2_spots.csv','B1_04_2_4_spots.csv']
F1_list = []
RECALL_list = []
PREC_list = []
slovnik_new = []
for sl in poradi:
    seznam_obrazku = Ziskej_obr(os.path.join(os.getcwd(),slozka[sl]),'tif')
    adresar_ulozeni = os.path.join(os.getcwd(),slozka_ulozeni[sl])
    plocha_all_images = []
    Zkontroluj_slozku(slozka_ulozeni[sl])
    
# prvni sada pres 200 obrazku
# gaussian_dict = {1: (119.5, 66.55215713212026),
#                  2: (229.5, 108.03040016461054),
#                  3: (355.0, 128.7770050799727),
#                  4: (555.5, 250.12789650389897)}

# # trvalo 33 minut pro stanoveni PRES VSECHNO
# gaussian_dict = {1: (136.0, 72.15442564845513),
#                   2: (248.5, 114.58050103980209),
#                   3: (373.5, 141.51299840398866),
#                   4: (576.0, 248.95364679759305)}

# with open('POCTY_HAAR_' + slozka + '_area_' +'.txt', 'w') as file:

    for cislo_obr in range(0,len(seznam_obrazku)):
        print(f'Zpracovava se obrazek cislo: {cislo_obr}')
        # =============================================================================
        # APLIKACE ICA ze zacatku
        # =============================================================================
        image = cv2.imread(seznam_obrazku[cislo_obr],cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # Zobraz(image)
        # ica_img = predzprac_ica(image) #ICA cast
        ica_img = predzprac_ica_OK(image) #ICA cast
        # Zobraz(ica_img)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(30,30)) #20-50
        image_s_ICA = clahe.apply(ica_img)
        # Zobraz(image_s_ICA)
        image_s_ICA = equalize_histogram_16bit(image_s_ICA)
        # Zobraz(image_s_ICA)
        
        kernel_size = 3
        noise_variance_ica = 68 #88 # Hodnota variance šumu (může být empiricky stanovena nebo odhadnuta) #95
        filtered_image_ica = wiener_filter(image_s_ICA, kernel_size, noise_variance_ica)
        # Zobraz(filtered_image_ica)
           
        # POZOR NACITAME UPRAVENY VSTUP PO VSECH FILTRECH - ICA-CLAHE-EKVALIZACE-WIENER PAK TO JDE SEMKA
        # Nastavení velikosti okna pro Haarovy funkce
        window_size = 3
        
        # Výpočet PP obrazu
        pp_image = compute_pp_image(filtered_image_ica, window_size)
        
        # Uložení výsledného obrazu
        totok = (pp_image*255).astype(np.uint8)
        # Zobraz(totok)
        
        # Aplikuj morfologickou dilataci pro uzavření mezer v konturách
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(totok, kernel, iterations=1) # 1-3 iterace by sly
        
        # Aplikuj morfologickou erozi, abys obnovil původní šířku kontury
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Najdi kontury na výsledném obrázku
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Vytvoř masku pro vyplnění kontur
        filled_image = np.zeros_like(totok)
        
        # Vyplň nalezené kontury
        cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)
            
        # Najdi kontury na vyplněné masce (nyní to jsou oblasti v masce)
        mask_contours, _ = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Zobraz(filled_image)
        # ************************************KONEC ZAKLADNI PROCES********************************************************** 
        
        # # Nastav minimální a maximální plochu pro filtrování kontur
        min_plocha = 70  # Nastavte podle potřeby #25
        max_plocha = 5000  # Nastavte podle potřeby
        
        # np.savez(os.path.join(slozka_ulozeni,"Bright_Field_tuple_data_" + str(cislo_obr+1) + ".npz"), *mask_contours)
        
        # Vyfiltruj kontury na základě plochy
        filtered_contours = [contour for contour in mask_contours if min_plocha <= cv2.contourArea(contour) <= max_plocha]       
        data_list = Nacti_souradnice(csv_soubor[sl],cislo_obr)
        
        metrics,test_obr = vyhodnot_kontury(filtered_contours, data_list,image,tolerance=5)
        
        # Nakresli_znacky_stred(test_obr, data_list)
        # Zobraz(test_obr)
        # Součet TP, FP, FN přes celou sadu
        
        F1_list.append(metrics['F1 Score'])
        RECALL_list.append(metrics['Recall'])
        PREC_list.append(metrics['Precision'])
            
        print(f"Precision: {metrics['Precision']:.2f}")
        print(f"Recall: {metrics['Recall']:.2f}")
        print(f"F1 Score: {metrics['F1 Score']:.2f}")
        print(f"True Positives: {metrics['TP']}")
        print(f"False Positives: {metrics['FP']}")
        print(f"False Negatives: {metrics['FN']}")
        
        # cv2.imshow('debil',test_obr)
        # cv2.imwrite(os.path.join(slozka_ulozeni[sl],"F1_score_overeni_kontury" + str(cislo_obr+1) + ".png"),test_obr)
        
        # slovnik_new = nadelej_slovnik_kontury(filtered_contours, data_list, tolerance=5)
    print('*****Nasleduje nova sada*****')   


#min z F1 0.8995403808273145
#max z F1 0.9576427255985267
plt.hist(F1_list, bins=25, edgecolor="black")  # bins nastav podle detailnosti
plt.xlabel("F1-Score")
plt.ylabel("Frequency")
# plt.title("Histogram of F1 Scores")
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.savefig("f1_score_histogram.pdf", dpi=300, bbox_inches="tight")
# plt.show()
plt.close()


#min z Recall 0.8896103896103896
#max z Recall 0.9616788321167883
plt.hist(RECALL_list, bins=25, edgecolor="black")  # bins nastav podle detailnosti
plt.xlabel("Recall")
plt.ylabel("Frequency")
# plt.title("Histogram of Recalls")
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.savefig("recall_histogram.pdf", dpi=300, bbox_inches="tight")
# plt.show()
plt.close()

#min z Precision 0.9061277705345502
#max z Precision 0.9737827715355806
plt.hist(PREC_list, bins=25, edgecolor="black")  # bins nastav podle detailnosti
plt.xlabel("Precision")
plt.ylabel("Frequency")
# plt.title("Histogram of Precisions")
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.savefig("precision_histogram.pdf", dpi=300, bbox_inches="tight")
# plt.show()
plt.close()


# total_tp = sum(tp_list)
# total_fp = sum(fp_list)
# total_fn = sum(fn_list)

# final_tp = [115541,119268]  
# final_fp = [6568,7069]
# final_fn = [8389,9016]

# # Celkové součty přes všechny sady
# total_tp_all = sum(final_tp)
# total_fp_all = sum(final_fp)
# total_fn_all = sum(final_fn)

# # Výpočet celkových metrik
# precision_all = total_tp_all / (total_tp_all + total_fp_all) if total_tp_all + total_fp_all > 0 else 0
# recall_all = total_tp_all / (total_tp_all + total_fn_all) if total_tp_all + total_fn_all > 0 else 0
# f1_score_all = (2 * precision_all * recall_all) / (precision_all + recall_all) if precision_all + recall_all > 0 else 0

# print(f"Celkové Precision: {precision_all:.2f}")
# print(f"Celkové Recall: {recall_all:.2f}")
# print(f"Celkové F1 Score: {f1_score_all:.2f}")

# F1 = 0.94
# Recall = 0.93
# Precision = 0.95

# PODSUD OK

# ***************************************************************************************
# =============================================================================
# # bez ICA
# # TOTAL_tp = sum([110134, 115734])
# # TOTAL_fp = sum([11259, 11171])
# # TOTAL_fn = sum([13796, 12550])
# # # # # # Výpočet celkových metrik
# # precision_all = TOTAL_tp / (TOTAL_tp + TOTAL_fp) if TOTAL_tp + TOTAL_fp > 0 else 0
# # recall_all = TOTAL_tp / (TOTAL_tp + TOTAL_fn) if TOTAL_tp + TOTAL_fn > 0 else 0
# # f1_score_all = (2 * precision_all * recall_all) / (precision_all + recall_all) if precision_all + recall_all > 0 else 0
# # # 
# # print(f"Celkové Precision: {precision_all:.2f}")
# # print(f"Celkové Recall: {recall_all:.2f}")
# # print(f"Celkové F1 Score: {f1_score_all:.2f}")
# # Celkové Precision: 0.91
# # Celkové Recall: 0.90
# # Celkové F1 Score: 0.90
# =============================================================================
# =============================================================================
# #  SE VSIM PREDZPRACOVANIM
# # TOTAL_tp = sum([115541, 119268])
# # TOTAL_fp = sum([6568, 7069])
# # TOTAL_fn = sum([8389, 9016])
# 
# # # # Výpočet celkových metrik
# # precision_all = TOTAL_tp / (TOTAL_tp + TOTAL_fp) if TOTAL_tp + TOTAL_fp > 0 else 0
# # recall_all = TOTAL_tp / (TOTAL_tp + TOTAL_fn) if TOTAL_tp + TOTAL_fn > 0 else 0
# # f1_score_all = (2 * precision_all * recall_all) / (precision_all + recall_all) if precision_all + recall_all > 0 else 0
# 
# # print(f"Celkové Precision: {precision_all:.2f}")
# # print(f"Celkové Recall: {recall_all:.2f}")
# # print(f"Celkové F1 Score: {f1_score_all:.2f}")

# # # Celkové Precision: 0.95
# # # Celkové Recall: 0.93
# # # Celkové F1 Score: 0.94
# =============================================================================

    

    # test_obr =  kresli_obarvene_kontury_gauss(filtered_contours, image,gaussian_dict,slovnik_new)
# print(f'Nejvetsi rozdil v sade byl {top_rodil}')

    # Nakresli_znacky_stred(test_obr, data_list)
    # Zobraz(test_obr)
    # test_obr = cv2.cvtColor(test_obr, cv2.COLOR_BGR2RGB)
    # Zobraz(test_obr)
    # cv2.imwrite(os.path.join(slozka_ulozeni,"Gauss_bright_Field_kontury_barvy" + str(cislo_obr+1) + ".png"),test_obr)
    
    

    # Nakresli_znacky_stred(test_img, data_list)
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(slozka_ulozeni,"Gauss_bright_Field_kontury_barvy" + str(cislo_obr+1) + ".png"),test_img)
    
    # slovnik_new.append(nadelej_slovnik_kontury(filtered_contours, data_list, tolerance=5))
       
    # # Vytvoření parametrů pro každý klíč
    # gaussian_dict = {}
    # for key, values in slovnik_new.items():
    #     gaussian_dict[key] = gaussian_params(values) #median a std dev
    
    # result_key, result_probs = classify_value(35, gaussian_dict)  
    # # testovaci na jeden obrazek
    
    # # Spojení všech hodnot do jednoho seznamu
    # velky_seznam = [hodnota for hodnoty in slovnik_new.values() for hodnota in hodnoty]
    # pocitadlo = 0
    # for hodnota in velky_seznam:
    #     vysl,_ = classify_value(hodnota, gaussian_dict)
    #     if vysl == 1:
    #         pocitadlo += 1
    #     elif vysl == 2:
    #         pocitadlo += 2
    #     elif vysl == 3:
    #         pocitadlo += 3
    #     else:
    #         pocitadlo += 4

# =============================================================================
# # Vytvoření slovníku seznamů podle klíčů pro danou sadu prozatim
# result_slouceni = {}
# for slovnik in slovnik_new:
#     for key, value in slovnik.items():
#         if key not in result_slouceni:
#             result_slouceni[key] = []  # Inicializace seznamu pro klíč
#         result_slouceni[key].extend(value)  # Přidání hodnot z aktuálního slovníku

# plt.figure(figsize=(10, 6))
# plt.hist(result_slouceni[4], bins=30, color='skyblue', edgecolor='black')
# plt.title("Histogram ploch kontur 4")
# plt.xlabel("Plocha kontury")
# plt.ylabel("Počet výskytů")
# # plt.yscale('log')  # Logaritmická stupnice pro lepší přehled
# plt.show()    
# =============================================================================
  

  
    
# =============================================================================
# # gaussian_dict = {}
# # for key, values in result_slouceni.items():
# #     gaussian_dict[key] = gaussian_params(values) #median a std dev        
# =============================================================================
        
        
        
    # print(f'Sectene kontury pomoci histogramu a gaussiana: {pocitadlo} Datalist pocet: {len(data_list)}')        

    # plt.figure(figsize=(10, 6))
    # plt.hist(result_slouceni[4], bins=30, color='skyblue', edgecolor='black')
    # plt.title("Histogram ploch kontur 4")
    # plt.xlabel("Plocha kontury")
    # plt.ylabel("Počet výskytů")
    # # plt.yscale('log')  # Logaritmická stupnice pro lepší přehled
    # plt.show()
    
    

    # metrics = vyhodnot_kontury(filtered_contours, data_list,tolerance=-5)    
    # Součet TP, FP, FN přes celou sadu
    # tp_list.append(metrics['TP'])
    # fp_list.append(metrics['FP'])
    # fn_list.append(metrics['FN'])
        
    # print(f"Precision: {metrics['Precision']:.2f}")
    # print(f"Recall: {metrics['Recall']:.2f}")
    # print(f"F1 Score: {metrics['F1 Score']:.2f}")
    # print(f"True Positives: {metrics['TP']}")
    # print(f"False Positives: {metrics['FP']}")
    # print(f"False Negatives: {metrics['FN']}")
    

    
    # Vykresleni filtrovanych ploch NA ORIGINALE - MENI JAS bezduvodne
    
    # background_image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    ## cv2.drawContours(background_image_rgb, filtered_contours, -1, 30000, 1) #zaloha  # Bílá barva s tloušťkou 2 v 16bit obraze
    # cv2.drawContours(background_image_rgb, filtered_contours, -1, 30000, 1)  # Bílá barva s tloušťkou 2 v 16bit obraze
    # Nakresli_znacky_stred(background_image_rgb, data_list)
    
    # cv2.imwrite(os.path.join(slozka_ulozeni,"Bright_Field_Maska_orig_" + str(cislo_obr+1) + ".png"),background_image_rgb)
    # Zobraz_RGB(background_image_rgb)

# =============================================================================
#     # nacteni tuplaka 
#     # data = np.load(os.path.join(slozka_ulozeni,"Bright_Field_tuple_data_" + str(cislo_obr+1) + ".npz"))
#     # loaded_tuple = tuple(data[f'arr_{i}'] for i in range(len(data.files)))
# =============================================================================



# =============================================================================
# samostatny vypocet metrik - rucni prace
# =============================================================================
# # Výpočet metrik
# total_tp = sum(tp_list)
# total_fp = sum(fp_list)
# total_fn = sum(fn_list)
# precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
# recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
# f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0    

# print(f"Precision pro sadu: {precision:.2f}")
# print(f"Recall pro sadu: {recall:.2f}")
# print(f"F1 Score pro sadu: {f1_score:.2f}")    

# final_tp = [277744,121691,258337,125303]  
# final_fp = [31213,22868,30142,28113]
# final_fn = [14358,2239,11272,2981]

# # Celkové součty přes všechny sady
# total_tp_all = sum(final_tp)
# total_fp_all = sum(final_fp)
# total_fn_all = sum(final_fn)

# # Výpočet celkových metrik
# precision_all = total_tp_all / (total_tp_all + total_fp_all) if total_tp_all + total_fp_all > 0 else 0
# recall_all = total_tp_all / (total_tp_all + total_fn_all) if total_tp_all + total_fn_all > 0 else 0
# f1_score_all = (2 * precision_all * recall_all) / (precision_all + recall_all) if precision_all + recall_all > 0 else 0

# print(f"Celkové Precision: {precision_all:.2f}")
# print(f"Celkové Recall: {recall_all:.2f}")
# print(f"Celkové F1 Score: {f1_score_all:.2f}")