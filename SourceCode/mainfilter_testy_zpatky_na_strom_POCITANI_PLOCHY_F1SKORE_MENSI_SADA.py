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

# Načtení obrázku TIFF
# soubor = r"D:\Projekty\GIT\Bunky_Marketa\B1_04_1_1_Bright_field\B1_04_1_1_Bright Field_001.tif" #1.Bright, ktomu 1. DAPi + 1. csv
# soubor = r"D:\Projekty\GIT\Bunky_Marketa\B1_04_1_2_Bright_field\B1_04_1_2_Bright Field_001.tif" #2.Bright, ktomu 1. DAPi + 2. csv
# soubor = r"D:\Projekty\GIT\Bunky_Marketa\B1_04_1_3_Bright_field\B1_04_1_3_Bright Field_001.tif" #3.Bright, ktomu 1. DAPi + 3. csv
# soubor = r"D:\Projekty\GIT\Bunky_Marketa\B1_04_1_4_Bright_field\B1_04_1_4_Bright Field_001.tif" #4.Bright, ktomu 1. DAPi + 4. csv

# soubor = r"D:\Projekty\GIT\Bunky_Marketa\B1_04_2_1_DAPI\B1_04_2_1_DAPI_001.tif" #1. DAPI 
# soubor = r"D:\Projekty\GIT\Bunky_Marketa\B1_04_2_2_DAPI\B1_04_2_2_DAPI_001.tif" #2. DAPI
# soubor = r"D:\Projekty\GIT\Bunky_Marketa\B1_04_2_3_DAPI\B1_04_2_3_DAPI_001.tif" #3. DAPI
# soubor = r"D:\Projekty\GIT\Bunky_Marketa\B1_04_2_4_DAPI\B1_04_2_4_DAPI_001.tif" #4. DAPI
top_rodil = 0

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
                cv2.ellipse(image,center_coor,axe_len,angle,startAngle,endAngle,color=(0,255,0),thickness=1)
    # cv2.imshow('Filtrovany obrazek + znacky', image)

def Nakresli_znacky_stred(image,data_list):
    # cv2.imshow('Filtrovany obrazek bez znacek', image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for pt in data_list:
                a, b, r = round(float(pt[4])), round(float(pt[5])), round(float(pt[9]))
                cv2.circle(image, (a, b), radius=1, color=(0, 0, 255), thickness=2) #tecky

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

def kresli_obarvene_kontury(contours, data_list, image, tolerance=5):
    """
    Funkce obarví kontury podle počtu buněk detekovaných uvnitř kontury.

    Parameters:
    - contours: Seznam kontur.
    - data_list: Seznam souřadnic bodů, kde každá položka má 31 hodnot. Souřadnice X a Y jsou na pozici 4 a 5.
    - image: Vstupní obraz, na který se kontury vykreslí.

    Returns:
    - image_with_contours: Obraz s obarvenými konturami.
    """
    jedna = 0
    dva = 0
    tri = 0
    ctyri = 0
    nebrat = 0
    # Převod 16bitového obrázku na 8bitový pro vykreslování
    image_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

    # Převod na RGB, aby bylo možné vykreslovat barevné kontury
    image_with_contours = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        bunka_in_contour = 0  # Počítadlo buněk v aktuální kontuře

        # Získáme bounding box pro aktuální konturu
        x, y, w, h = cv2.boundingRect(contour)

        # Projdeme všechny souřadnice v data_list
        for data in data_list:
            px = round(float(data[4]))
            py = round(float(data[5]))

            # Kontrola, zda souřadnice spadají do bounding boxu a jsou uvnitř kontury
            # if x <= px <= x + w and y <= py <= y + h:
            if (x - tolerance) <= px <= (x+w+tolerance) and (y - tolerance) <= py <= (y+h+tolerance):
                # Kontrola, zda bod leží uvnitř kontury nebo v toleranční vzdálenosti od kontury
                distance = cv2.pointPolygonTest(contour,(px,py),True)
                if -tolerance <= distance <= tolerance: # Uvnitř kontury nebo v rámci tolerance
                # if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                    bunka_in_contour += 1

        # Vybereme barvu podle počtu buněk
        
        if bunka_in_contour == 1:
             color = (255, 0, 0)  # Červená
             cv2.drawContours(image_with_contours, [contour], -1, color, 1)
             jedna += 1
        elif bunka_in_contour < 1:
            color = (0,0,0)
            cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            nebrat += 1
        elif bunka_in_contour == 2:
             color = (0, 0, 255)  # Modrá
             cv2.drawContours(image_with_contours, [contour], -1, color, 1)
             dva +=1
        elif bunka_in_contour == 3:
            color = (0, 255, 0)  # Zelená
            cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            tri += 1
        elif bunka_in_contour > 3:
            color = (255, 255, 0)  # Žlutá pro více než 3 buňky
            cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            ctyri += 1

    # Výstupy
    print(f'Suma platnych kontur: {jedna + dva*2 + tri*3 + ctyri*4}')
    print(f'Pocet bunek dle data_list: {len(data_list)}')
    # print(f'Počet kontur s 1 buňkami: {jedna}')
    # print(f'Počet kontur s 2 buňkami: {dva}')
    # print(f'Počet kontur s 3 buňkami: {tri}')
    # print(f'Počet kontur s 4 buňkami: {ctyri}')
    print(f'Počet kontur bez bunek: {nebrat}')    
    return image_with_contours

def vyhodnot_kontury(contours, data_list, tolerance=-5):
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
#*************************************************************************************************************
def nadelej_slovnik_kontury(contours, data_list, tolerance=5):
    """
    Funkce obarví kontury podle počtu buněk detekovaných uvnitř kontury.

    Parameters:
    - contours: Seznam kontur.
    - data_list: Seznam souřadnic bodů, kde každá položka má 31 hodnot. Souřadnice X a Y jsou na pozici 4 a 5.
    - image: Vstupní obraz, na který se kontury vykreslí.

    Returns:
    - image_with_contours: Obraz s obarvenými konturami.
    """
    jedna = 0
    dva = 0
    tri = 0
    ctyri = 0
    nebrat = 0
    slovnik_kontur = {}
    
    # Převod 16bitového obrázku na 8bitový pro vykreslování
    # image_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

    # Převod na RGB, aby bylo možné vykreslovat barevné kontury
    # image_with_contours = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        bunka_in_contour = 0  # Počítadlo buněk v aktuální kontuře

        # Získáme bounding box pro aktuální konturu
        x, y, w, h = cv2.boundingRect(contour)

        # Projdeme všechny souřadnice v data_list
        for data in data_list:
            px = round(float(data[4]))
            py = round(float(data[5]))

            # Kontrola, zda souřadnice spadají do bounding boxu a jsou uvnitř kontury
            # if x <= px <= x + w and y <= py <= y + h:
            if (x - tolerance) <= px <= (x+w+tolerance) and (y - tolerance) <= py <= (y+h+tolerance):
                # Kontrola, zda bod leží uvnitř kontury nebo v toleranční vzdálenosti od kontury
                distance = cv2.pointPolygonTest(contour,(px,py),True)
                if -tolerance <= distance <= tolerance: # Uvnitř kontury nebo v rámci tolerance
                # if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                    bunka_in_contour += 1

        # Vybereme barvu podle počtu buněk
        
        if bunka_in_contour == 1:
             color = (255, 0, 0)  # Červená
             # cv2.drawContours(image_with_contours, [contour], -1, color, 1)
             jedna += 1             
             # Pokud hodnota není seznam, převeď ji na seznam
             if not isinstance(slovnik_kontur.get(1), list):
                 slovnik_kontur[1] = [slovnik_kontur[1]] if 1 in slovnik_kontur else []   
             slovnik_kontur[1].append(cv2.contourArea(contour))
        elif bunka_in_contour < 1:
            color = (0,0,0)
            # cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            nebrat += 1
            # # Pokud hodnota není seznam, převeď ji na seznam
            # if not isinstance(slovnik_kontur.get(0), list):
            #     slovnik_kontur[0] = [slovnik_kontur[0]] if 0 in slovnik_kontur else []   
            # slovnik_kontur[0].append(cv2.contourArea(contour))
        elif bunka_in_contour == 2:
             color = (0, 0, 255)  # Modrá
             # cv2.drawContours(image_with_contours, [contour], -1, color, 1)
             dva +=1
             # Pokud hodnota není seznam, převeď ji na seznam
             if not isinstance(slovnik_kontur.get(2), list):
                 slovnik_kontur[2] = [slovnik_kontur[2]] if 2 in slovnik_kontur else []   
             slovnik_kontur[2].append(cv2.contourArea(contour))
        elif bunka_in_contour == 3:
            color = (0, 255, 0)  # Zelená
            # cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            tri += 1
            # Pokud hodnota není seznam, převeď ji na seznam
            if not isinstance(slovnik_kontur.get(3), list):
                slovnik_kontur[3] = [slovnik_kontur[3]] if 3 in slovnik_kontur else []   
            slovnik_kontur[3].append(cv2.contourArea(contour))
        elif bunka_in_contour > 3:
            color = (255, 255, 0)  # Žlutá pro více než 3 buňky
            # cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            ctyri += 1
            # Pokud hodnota není seznam, převeď ji na seznam
            if not isinstance(slovnik_kontur.get(4), list):
                slovnik_kontur[4] = [slovnik_kontur[4]] if 4 in slovnik_kontur else []   
            slovnik_kontur[4].append(cv2.contourArea(contour))

        # Vykreslení kontury na obrázek
        # cv2.drawContours(image_with_contours, [contour], -1, color, 2)
    # Výstupy
    print(f'Suma platnych bunek konturach: {jedna + dva*2 + tri*3 + ctyri*4}')
    print(f'Pocet bunek dle data_list: {len(data_list)}')
    # print(f'Počet kontur s 1 buňkami: {jedna}')
    # print(f'Počet kontur s 2 buňkami: {dva}')
    # print(f'Počet kontur s 3 buňkami: {tri}')
    # print(f'Počet kontur s 4 buňkami: {ctyri}')
    # print(f'Počet kontur bez bunek: {nebrat}')    
    return slovnik_kontur


def gaussian_params(values):
    values = np.array(values)
    median = np.median(values)
    std_dev = np.std(values)  # Standardní odchylka jako šířka křivky
    return median, std_dev

# Funkce pro výpočet pravděpodobnosti z Gaussovy křivky
def gaussian_probability(x, median, std_dev):
    # Vzorec pro normální rozdělení
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - median) / std_dev) ** 2)

def gaussian_probability_normalized(x, median, std_dev):
    # Vzorec pro normální rozdělení
    raw_probability = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - median) / std_dev) ** 2)
    # Normalizace pravděpodobnosti
    max_value = 1 / (std_dev * np.sqrt(2 * np.pi))
    normalized_probability = raw_probability / max_value
    return normalized_probability


# # pro MENSI SADY 400 obrazku TRVALO 12 MINUT
# gaussian_dict = {1: (150, 80.67077528531642),
#                   2: (274, 124.45749864666038),
#                   3: (411, 153.28225307006574),
#                   4: (628.5, 260.34850615489836)}

# Porovnání pravděpodobností pro danou hodnotu
def classify_value(x, gaussian_dict,cutoff_left=None):
    probabilities = {}
    for key, (median, std_dev) in gaussian_dict.items():
        if key == 1 and cutoff_left is not None and x < cutoff_left:
            best_key = 0
            # print(f'Ignorovana kontury s plochou {x}')
            return best_key,None
        else:
            # probabilities[key] = gaussian_probability(x, median, std_dev)
            probabilities[key] = gaussian_probability_normalized(x, median, std_dev)
    

# =============================================================================
#     if (x > gaussian_dict[2][0] and (x >= gaussian_dict[2][0] + 0.5*gaussian_dict[2][1])): #zkusme vyresit problem, kdyz je to na hrane mezi 2 a 3, uprednostnime 2
#         best_key = 3
#         return best_key,None
# =============================================================================

# 0.35 prob 1 -2  
# 0.25 prob 2 -3 
    if (x > gaussian_dict[1][0] and (x <=270) and (abs(probabilities[1] - probabilities[2])) <= 0.35): #zkusme vyresit problem, kdyz je to na hrane mezi 1 a 2, uprednostnime 1
        best_key = 1
        return best_key,None
    if (x > gaussian_dict[2][0] and (x <= 411) and (abs(probabilities[2] - probabilities[3])) <= 0.25): #zkusme vyresit problem, kdyz je to na hrane mezi 2 a 3, uprednostnime 2
        best_key = 2
        return best_key,None
        
    # Najít klíč s nejvyšší pravděpodobností
    # print(f'velikost kontury je: {x}')
    # print(f'pravdepodobnost 1: {probabilities[1]}')
    # print(f'pravdepodobnost 2: {probabilities[2]}')
    # print(f'pravdepodobnost 3: {probabilities[3]}')
    # print(f'pravdepodobnost 4: {probabilities[4]}')
    best_key = max(probabilities, key=probabilities.get)
    # print(f'nejlepsi pst: {best_key}')
    return best_key, probabilities


def kresli_obarvene_kontury_gauss(contours, image,gaussian_dict):
    global top_rodil
    """
    Funkce obarví kontury podle počtu buněk detekovaných uvnitř kontury. GAUSS

    Parameters:
    - contours: Seznam kontur.
    - data_list: Seznam souřadnic bodů, kde každá položka má 31 hodnot. Souřadnice X a Y jsou na pozici 4 a 5.
    - image: Vstupní obraz, na který se kontury vykreslí.

    Returns:
    - image_with_contours: Obraz s obarvenými konturami.
    """
    jedna = 0
    dva = 0
    tri = 0
    ctyri = 0
    nebrat = 0
    # Převod 16bitového obrázku na 8bitový pro vykreslování
    image_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

    # Převod na RGB, aby bylo možné vykreslovat barevné kontury
    image_with_contours = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
    
    # bunka_in_contour = 0  # Počítadlo buněk v aktuální kontuře
    for contour in contours:
        # pc_bunek,_ = classify_value(cv2.contourArea(contour),gaussian_dict,cutoff_left=40)
        pc_bunek,_ = classify_value(cv2.contourArea(contour),gaussian_dict)
        
        if pc_bunek == 1:
             color = (0, 0, 255)  # Červená
             cv2.drawContours(image_with_contours, [contour], -1, color, 1)
             jedna += 1
        elif pc_bunek < 1:
            color = (0,0,0)
            cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            nebrat += 1
        elif pc_bunek == 2:
             color = (255, 0, 0)  # Modrá
             cv2.drawContours(image_with_contours, [contour], -1, color, 1)
             dva +=1
        elif pc_bunek == 3:
            color = (0, 255, 0)  # Zelená
            cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            tri += 1
        elif pc_bunek > 3:
            color = (0, 255, 255)  # Žlutá pro více než 3 buňky
            cv2.drawContours(image_with_contours, [contour], -1, color, 1)
            tri += 1

    # Výstupy
    print(f'Pocet bunek dle data_list GAUSS: {len(data_list)}')
    print(f'Suma platnych bunek v konturach GAUSS: {jedna + dva*2 + tri*3 + ctyri*4}') #zkusme s 5 ve ctyri
    print(f'Rozdil oproti souradnicim s jadry a GAUSS: {abs((jedna + dva*2 + tri*3 + ctyri*4) - len(data_list))}')
    pom = abs((jedna + dva*2 + tri*3 + ctyri*4) - len(data_list))
    if pom > top_rodil:
        top_rodil = copy.deepcopy(pom)
    
    # print(f'Počet kontur s 1 buňkami: {jedna}')
    # print(f'Počet kontur s 2 buňkami: {dva}')
    # print(f'Počet kontur s 3 buňkami: {tri}')
    # print(f'Počet kontur s 4 buňkami: {ctyri}')
    # print(f'Počet kontur bez bunek: {nebrat}')    
    return image_with_contours


# def kresli_a_vyhodnot_kontury_GAUSS_POCET(contours, image, gaussian_dict, data_list, tolerance=-5):
#     """
#     Funkce obarví kontury podle počtu buněk detekovaných uvnitř kontury a vyhodnotí přesnost detekce.

#     Parameters:
#     - contours: Seznam kontur.
#     - image: Vstupní obraz, na který se kontury vykreslí.
#     - gaussian_dict: Slovník pro klasifikaci počtu buněk podle velikosti oblasti.
#     - data_list: Seznam středových souřadnic buněk (každá položka má hodnoty na indexech 4 a 5).
#     - tolerance: Tolerance pro určení, zda bod leží uvnitř kontury.

#     Returns:
#     - image_with_contours: Obraz s obarvenými konturami.
#     - metrics: Slovník s hodnotami TP, FP, FN, Precision, Recall a F1 skóre.
#     """
#     jedna, dva, tri, ctyri, nebrat = 0, 0, 0, 0, 0
#     tp, fp, fn = 0, 0, 0

#     # Převod 16bitového obrázku na 8bitový pro vykreslování
#     image_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

#     # Převod na RGB, aby bylo možné vykreslovat barevné kontury
#     image_with_contours = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)

#     # Převést středové souřadnice na seznam tuple (px, py)
#     centers = [(round(float(data[4])), round(float(data[5]))) for data in data_list]
#     detected_centers = [False] * len(centers)

#     for contour in contours:
#         pc_bunek, _ = classify_value(cv2.contourArea(contour), gaussian_dict)
#         found_center = False

#         # Kreslení kontur podle počtu buněk
#         if pc_bunek == 1:
#             color = (255, 0, 0)  # Červená
#             jedna += 1
#         elif pc_bunek < 1:
#             color = (0, 0, 0)  # Černá
#             nebrat += 1
#         elif pc_bunek == 2:
#             color = (0, 0, 255)  # Modrá
#             dva += 1
#         elif pc_bunek == 3:
#             color = (0, 255, 0)  # Zelená
#             tri += 1
#         elif pc_bunek > 3:
#             color = (255, 255, 0)  # Žlutá
#             ctyri += 1

#         cv2.drawContours(image_with_contours, [contour], -1, color, 1)

#         # Vyhodnocení přesnosti (True Positive, False Positive, False Negative)
#         for i, (px, py) in enumerate(centers):
#             if not detected_centers[i] and cv2.pointPolygonTest(contour, (px, py), True) >= tolerance:
#                 tp += 1
#                 detected_centers[i] = True
#                 found_center = True

#         if not found_center:
#             fp += 1

#     # Zbývající nepřiřazené středové body
#     fn = detected_centers.count(False)

#     # Výstup metrik
#     print(f"Suma platných buněk v konturách GAUSS: {jedna + dva*2 + tri*3 + ctyri*4}")
#     print(f"Rozdíl oproti souřadnicím s jádry: {abs((jedna + dva*2 + tri*3 + ctyri*4) - len(data_list))}")
#     # print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

#     metrics = {"TP": tp, "FP": fp, "FN": fn}
#     return image_with_contours, metrics

def kresli_a_vyhodnot_kontury_GAUSS_POCET(contours, image, gaussian_dict, data_list, tol=50,draw=False):
    """
    Funkce obarví kontury podle počtu buněk detekovaných uvnitř kontury a vyhodnotí přesnost detekce
    na základě součtu detekovaných buněk a porovnání s referenčním počtem buněk.

    Parameters:
    - contours: Seznam kontur.
    - image: Vstupní obraz, na který se kontury vykreslí.
    - gaussian_dict: Slovník pro klasifikaci počtu buněk podle velikosti oblasti.
    - data_list: Seznam středových souřadnic buněk (každá položka má hodnoty na indexech 4 a 5).
    - tolerance: Tolerance pro určení, zda bod leží uvnitř kontury.
    - tol: Toleranční interval ± pro počítání TP.

    Returns:
    - image_with_contours: Obraz s obarvenými konturami.
    - metrics: Slovník s hodnotami TP, FP, FN a dalšími statistikami.
    """
    jedna, dva, tri, ctyri, nebrat = 0, 0, 0, 0, 0

    # Převod 16bitového obrázku na 8bitový pro vykreslování
    if draw:
        image_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))
        image_with_contours = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
    else:
        image_with_contours = None

    # Odhad celkového počtu buněk na základě kontur
    total_detected = 0

    for contour in contours:
        pc_bunek, _ = classify_value(cv2.contourArea(contour), gaussian_dict)
        total_detected += pc_bunek

        # Kreslení kontur podle počtu buněk
        if pc_bunek == 1:
            color = (255, 0, 0)  # Červená
            jedna += 1
        elif pc_bunek < 1:
            color = (0, 0, 0)  # Černá
            nebrat += 1
        elif pc_bunek == 2:
            color = (0, 0, 255)  # Modrá
            dva += 1
        elif pc_bunek == 3:
            color = (0, 255, 0)  # Zelená
            tri += 1
        elif pc_bunek > 3:
            color = (255, 255, 0)  # Žlutá
            ctyri += 1
        
        if draw:    
            cv2.drawContours(image_with_contours, [contour], -1, color, 1)

    # Referenční hodnota (skutečný počet buněk)
    total_actual = len(data_list)

    # Vyhodnocení přesnosti
    tp = 0
    fp = 0
    fn = 0

    # Rozdíl mezi odhadem a skutečným počtem
    diff = total_detected - total_actual

    if abs(diff) <= tol:
        tp = 1  # Správný odhad v rámci tolerance
    else:
        if diff > 0:  # Odhad je vyšší než skutečný počet
            fp = 1
        else:  # Odhad je nižší než skutečný počet
            fn = 1


    # Výstup metrik
    print(f"Suma detekovaných buněk GAUSS: {total_detected}")
    print(f"Skutečný počet buněk (referenční): {total_actual}")
    print(f"Rozdíl: {abs(diff)}")
    # print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

    metrics = {
        # "Total Detected": total_detected,
        # "Total Actual": total_actual,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        # "Precision": precision,
        # "Recall": recall,
        # "F1 Score": f1_score,
    }

    return image_with_contours, metrics

#*********************************************************************************************************

# slozka = 'B1_04_1_1_Bright_field'    
slozka = 'B1_04_1_2_Bright_field'  # toto ofiko pouzivame
# slozka = 'B1_04_1_3_Bright_field'
# slozka = 'B1_04_1_4_Bright_field'   # toto ofiko pouzivame



# slozka_ulozeni = 'Gauss_bright_Field_kontury_barvy_B1_04_1_1'
slozka_ulozeni = 'Gauss_bright_Field_kontury_barvy_B1_04_1_2'  # toto ofiko pouzivame
# slozka_ulozeni = 'Gauss_bright_Field_kontury_barvy_B1_04_1_3'
# slozka_ulozeni = 'Gauss_bright_Field_kontury_barvy_B1_04_1_4'  # toto ofiko pouzivame

# slozka_ulozeni = 'B1_04_1_2_Haar'
# slozka_ulozeni = 'B1_04_1_4_Haar'


# csv_soubor = 'B1_04_2_1_spots.csv'
csv_soubor = 'B1_04_2_2_spots.csv'  # toto ofiko pouzivame
# csv_soubor = 'B1_04_2_3_spots.csv'
# csv_soubor = 'B1_04_2_4_spots.csv'   # toto ofiko pouzivame


seznam_obrazku = Ziskej_obr(os.path.join(os.getcwd(),slozka),'tif')
# seznam_obrazku = Ziskej_obr(os.path.join(os.getcwd(),slozka),'tif')

adresar_ulozeni = os.path.join(os.getcwd(),slozka_ulozeni)
plocha_all_images = []

tp_list = []
fp_list = []
fn_list = []
slovnik_new = []

# pro MENSI SADY 400 obrazku TRVALO 12 MINUT
gaussian_dict = {1: (150, 80.67077528531642),
                  2: (274, 124.45749864666038),
                  3: (411, 153.28225307006574),
                  4: (628.5, 260.34850615489836)}

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
    noise_variance_ica = 68 #bylo posledni 68 #88 # Hodnota variance šumu (může být empiricky stanovena nebo odhadnuta) #95
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
    # Zobraz(eroded)
    
    # Najdi kontury na výsledném obrázku
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Vytvoř masku pro vyplnění kontur
    filled_image = np.zeros_like(totok)
    
    # Vyplň nalezené kontury
    cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)
    # Zobraz(filled_image)
    
    # Najdi kontury na vyplněné masce (nyní to jsou oblasti v masce)
    mask_contours, _ = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Zobraz(filled_image)
    # ************************************KONEC ZAKLADNI PROCES********************************************************** 
    
    # Zde ty masky
# =============================================================================
    
    # # Pokud má maska pouze jeden kanál, převedeme ji na RGB (tři kanály)
    # mask_rgb = cv2.cvtColor(filled_image, cv2.COLOR_GRAY2BGR)
    
    # # Nastav červenou barvu (v RGB) do oblastí s maskou (kde je maska 255)
    # # mask_rgb[filled_image == 255] = [0, 0, 255]  # Červená barva
    # mask_rgb[filled_image == 255] = [255, 0, 0]  # Červená barva
    
    # # Konvertuj 16bitový jednokanálový obrázek na tříkanálový (RGB)
    # background_image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # # Připrav alfa blending (smíchání s průhledností)
    # alpha = 0.1  # Nastav průhlednost (0.0 = zcela průhledné, 1.0 = zcela neprůhledné)
    
    # # Převeď masku na 16bitovou úroveň (aby odpovídala 16bitovému pozadí)
    # mask_rgb_16bit = np.uint16(mask_rgb) * 256  # Zvětší hodnoty na 16bitovou úroveň (65535 pro bílou)
    
    # # Vytvoř binární masku tam, kde je červená barva
    # # red_mask = (mask_rgb[:, :, 2] == 255)  # Identifikuj červené oblasti
    # red_mask = (mask_rgb[:, :, 0] == 255)  # Identifikuj červené oblasti
    
    # blended_image = background_image_rgb.copy()
    # blended_image[red_mask] = cv2.addWeighted(background_image_rgb[red_mask], 1 - alpha, mask_rgb_16bit[red_mask], alpha, 0, dtype=cv2.CV_16U)
    
    # data_list = Nacti_souradnice(csv_soubor,cislo_obr)
    # Nakresli_znacky(blended_image,data_list) #toto
    # Zobraz_RGB(blended_image)
    # # cv2.imwrite(os.path.join(slozka_ulozeni,"Bright_Field_ica_" + str(cislo_obr+1) + ".tif"),blended_image)
# =============================================================================
    
    
    
    # # Nastav minimální a maximální plochu pro filtrování kontur
    min_plocha = 70  # Nastavte podle potřeby #25 bylo naladeno na 45
    max_plocha = 5000  # Nastavte podle potřeby
    
    # np.savez(os.path.join(slozka_ulozeni,"Bright_Field_tuple_data_" + str(cislo_obr+1) + ".npz"), *mask_contours)
    
    # Vyfiltruj kontury na základě plochy
    filtered_contours = [contour for contour in mask_contours if min_plocha <= cv2.contourArea(contour) <= max_plocha]#\BACHA POCITAM MAX 3 bunky v konture
    data_list = Nacti_souradnice(csv_soubor,cislo_obr)
    test_obr =  kresli_obarvene_kontury_gauss(filtered_contours, image,gaussian_dict)
    cv2.imwrite(os.path.join(slozka_ulozeni,'Gauss_kontury_final_barvy'+ '_obr_' + str(cislo_obr+1) + ".png"),test_obr) #BGR bere
    # Nakresli_znacky_stred(test_obr, data_list)
    # Zobraz(test_obr)
    
    # slovnik_new = nadelej_slovnik_kontury(filtered_contours, data_list, tolerance=5)
    
    
# print(f'Nejvetsi rozdil v sade byl {top_rodil}')
    
    # test_obr,metrics = kresli_a_vyhodnot_kontury_GAUSS_POCET(filtered_contours, image, gaussian_dict, data_list,tol=50,draw=True) #toto se pouzivalo
    # _, metrics = kresli_a_vyhodnot_kontury_gauss(filtered_contours,image, gaussian_dict, data_list, tolerance=-5)
    
    # # Součet TP, FP, FN přes celou sadu
    # tp_list.append(metrics['TP'])
    # fp_list.append(metrics['FP'])
    # fn_list.append(metrics['FN'])

# # # # Výpočet metrik
# total_tp = sum(tp_list)
# total_fp = sum(fp_list)
# total_fn = sum(fn_list)
# precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
# recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
# f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0    
# print(f"Precision pro sadu: {precision:.2f}")
# print(f"Recall pro sadu: {recall:.2f}")
# print(f"F1 Score pro sadu: {f1_score:.2f}")    
# final_tp = [187,159]  
# final_fp = [8,41]
# final_fn = [5,0]    

# ****************************************************
# vysl mensi sada 400 obr max 3 bunky v konture toto plati PRO VSECHNY BODY PREDZPRACOVANI
# Celkové Precision: 0.88
# Celkové Recall: 0.99
# Celkové F1 Score: 0.93
# ****************************************************




    # slovnik_new = nadelej_slovnik_kontury(filtered_contours, data_list, tolerance=5)
    
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

# Vytvorime si slovnik ze vsech hodnot
# # gaussian_dict = {}
# # for key, values in result_slouceni.items():
# #     gaussian_dict[key] = gaussian_params(values) #median a std dev        
# =============================================================================

# =============================================================================
  

  
    

        
        
    

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
    
    
    # # zkusme obarveni, jak to vypada
    # test_img = kresli_obarvene_kontury(filtered_contours, data_list, image,tolerance=5)
    # Nakresli_znacky_stred(test_img, data_list)
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(slozka_ulozeni,"Barvy_bright_Field_kontury_barvy" + str(cislo_obr+1) + ".png"),test_img)
    # Zobraz(test_img)
    # Zobraz_RGB(test_img)
    
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