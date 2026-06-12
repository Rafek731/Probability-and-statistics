import math
from typing import Any, Optional, Literal
from scipy.stats import norm, t, chi2


# ==========================================
# CZĘŚĆ I: Twierdzenia Graniczne (X_CTG.ipynb)
# ==========================================

def poisson(n: int, p: float, k: int) -> float:
    """
    Szacuje prawdopodobieństwo uzyskania dokładnie `k` sukcesów w schemacie Bernoulliego,
    wykorzystując przybliżenie z twierdzenia Poissona.

    Args:
        n (int): Liczba przeprowadzonych prób.
        p (float): Prawdopodobieństwo odniesienia sukcesu w pojedynczej próbie.
        k (int): Oczekiwana liczba sukcesów.

    Returns:
        float: Oszacowane prawdopodobieństwo uzyskania dokładnie `k` sukcesów.

    Example:
        >>> poisson(10000000, 0.00000007, 1)
        0.34760971265398666
    """
    lam = n * p
    return (lam**k) / math.factorial(k) * math.exp(-lam)


def poisson_with_error(n: int, p: float, ks: list[int]) -> tuple[float, float]:
    """
    Szacuje prawdopodobieństwo uzyskania liczby sukcesów ze zbioru `k` oraz błąd oszacowania.
    
    Wykorzystuje przybliżenie Poissona dla sumy prawdopodobieństw z podanej listy
    oczekiwanych ilości sukcesów w schemacie Bernoulliego.

    Args:
        n (int): Liczba przeprowadzonych prób.
        p (float): Prawdopodobieństwo odniesienia sukcesu w pojedynczej próbie.
        k (list[int]): lista interesujących nas ilości sukcesów (np. [0, 1, 2]).

    Returns:
        Tuple[float, float]: Krotka zawierająca:
            - Szacowane prawdopodobieństwo otrzymania liczby sukcesów podanej w liście.
            - Maksymalny błąd tego oszacowania.

    Example:
        >>> poisson_ext(10000000, 0.00000007, [0, 1, 2])
        (0.9658584158742916, 4.900000000000001e-08)
    """
    lam = n * p
    result = [poisson(n, p, k_int) for k_int in ks]
    return sum(result), (lam ** 2) / n


def CTG(n: int, mu: float, sigma: float, lower_lim: float = -math.inf, upper_lim: float = math.inf) -> float:
    """
    Szacuje prawdopodobieństwo na podstawie Centralnego Twierdzenia Granicznego (CTG).

    Oblicza prawdopodobieństwo tego, że suma `n` niezależnych zmiennych losowych
    o podanej wartości oczekiwanej i odchyleniu standardowym wpadnie w przedział [d, g].

    Args:
        n (int): Liczba niezależnych zmiennych losowych w sumie.
        mu (float): Wartość oczekiwana (średnia) pojedynczej zmiennej losowej.
        sigma (float): Odchylenie standardowe pojedynczej zmiennej losowej.
        lower_lim (float, optional): Dolne ograniczenie przedziału. Domyślnie -inf.
        upper_lim (float, optional): Górne ograniczenie przedziału. Domyślnie +inf.

    Returns:
        float: Szacowane prawdopodobieństwo znalezienia się sumy w przedziale [lower_lim, upper_lim].

    Example:
        >>> CTG(400, 0.3, math.sqrt(0.3 * 0.7), lower_lim=130)
        0.13761676203741713
    """
    return norm.cdf((upper_lim - n * mu) / (sigma * math.sqrt(n))) - norm.cdf((lower_lim - n * mu) / (sigma * math.sqrt(n)))


def rozmiar(p: float, d: float, pb: float) -> int:
    """
    Oblicza minimalny rozmiar próby wymagany do osiągnięcia zadanego błędu oszacowania.

    Args:
        p (float): Ograniczenie na prawdopodobieństwo uzyskania sukcesu.
        d (float): Rozmiar dopuszczalnego odchylenia (margines błędu).
        pb (float): Prawdopodobieństwo, z jakim uzyskamy błąd (poziom istotności).

    Returns:
        int: Minimalny rozmiar próby (zaokrąglony w górę do pełnej liczby całkowitej).

    Example:
        >>> rozmiar(0.5, 3.375, 0.5)
        101
    """
    return math.ceil(d ** 2 / ((norm.ppf((2 - pb) / 2) ** 2) * p * (1 - p)))


# ==========================================
# CZĘŚĆ II: Estymatory (XI_Estymatory.ipynb)
# ==========================================

def no_estymator_wariancji(dane: list[float]) -> float:
    """
    Oblicza nieobciążony estymator wariancji dla podanej próby.

    Args:
        dane (list[float]): lista zawierająca wyniki pobrane z próby.

    Returns:
        float: Wartość nieobciążonego estymatora wariancji.

    Example:
        >>> NEW([1, 2, 3, 4, 5, 6, 7, 8, 9])
        7.5
    """
    n = len(dane)
    x_bar = sum(dane) / n
    sq_diff = list(map(lambda x: (x - x_bar)**2, dane))
    return sum(sq_diff) / (n - 1)


def przedzial_ufnosci_srednia(dane: list[float], odch_standardowe: float, alfa: float) -> tuple[float, float]:
    """
    Wyznacza przedział ufności dla wartości średniej przy znanym odchyleniu standardowym populacji.

    Args:
        dane (list[float]): lista zawierająca wyniki pobrane z próby.
        od_st (float): Znane odchylenie standardowe w populacji.
        alfa (float): Poziom błędu (np. 0.05 dla poziomu ufności 95%).

    Returns:
        Tuple[float, float]: Krotka zawierająca dolną i górną granicę przedziału ufności.

    Example:
        >>> esty_sr([2, 2.3, 2.4, 2.5, 1.9, 2.3, 2.5, 2.1, 2.4, 2.3], 1, 0.05)
        (1.6502049676954385, 2.8897950323045616)
    """
    n = len(dane)
    x_bar = sum(dane) / n
    r = norm.ppf(1 - alfa / 2) * odch_standardowe / math.sqrt(n)
    return float(x_bar - r), float(x_bar + r)


def przedzial_ufnosci_srednia_bez_odch_stand(dane: list[float], alfa: float) -> tuple[float, float]:
    """
    Wyznacza przedział ufności dla wartości średniej przy nieznanym odchyleniu standardowym populacji.
    
    Automatycznie dobiera odpowiedni rozkład w zależności od rozmiaru próby:
    - Dla małych prób (n <= 30) korzysta z rozkładu t-Studenta.
    - Dla dużych prób (n > 30) korzysta ze standardowego rozkładu normalnego.

    Args:
        dane (list[float]): lista zawierająca wyniki pobrane z próby.
        alfa (float): Poziom błędu (np. 0.05 dla poziomu ufności 95%).

    Returns:
        Tuple[float, float]: Krotka zawierająca dolną i górną granicę przedziału ufności.

    Example:
        >>> esty_sr_bez_od([2, 2.3, 2.4, 2.5, 1.9, 2.3, 2.5, 2.1, 2.4, 2.3], 0.05)
        (2.1228148457808524, 2.4171851542191476)
    """
    n = len(dane)
    x_bar = sum(dane) / n
    s = math.sqrt(no_estymator_wariancji(dane))

    if n <= 30:
        dist = t.ppf(1 - alfa / 2, n - 1)
    else:
        dist = norm.ppf(1 - alfa / 2)

    r = dist * s / math.sqrt(n)
    
    return float(x_bar - r), float(x_bar + r)


def przedzial_ufnosci_wariancja(dane: list[float], alfa: float) -> tuple[float, float]:
    """
    Wyznacza przedział ufności dla wariancji z wykorzystaniem rozkładu chi-kwadrat.

    Args:
        dane (list[float]): lista zawierająca wyniki pobrane z próby.
        alfa (float): Poziom błędu (np. 0.05 dla poziomu ufności 95%).

    Returns:
        Tuple[float, float]: Krotka zawierająca dolną i górną granicę przedziału ufności dla wariancji.

    Example:
        >>> esty_war([2, 2.3, 2.4, 2.5, 1.9, 2.3, 2.5, 2.1, 2.4, 2.3], 0.05)
        (0.020028631166238927, 0.14109075746397742)
    """
    n = len(dane)
    numerator = (n - 1) * no_estymator_wariancji(dane)
    return float(numerator / chi2.ppf(1 - alfa / 2, n - 1)), float(numerator / chi2.ppf(alfa / 2, n - 1))


def przedzial_ufnosci_proporcja(lista: list[Any], wart: Any, alfa: float) -> tuple[float, float]:
    """
    Wyznacza przedział ufności dla proporcji (częstości występowania danej wartości w populacji).

    Args:
        lista (list[Any]): lista zawierająca wyniki pobrane z próby (np. zbiór wylosowanych kolorów).
        wart (Any): Konkretna wartość, której proporcję chcemy estymować.
        alfa (float): Poziom błędu (np. 0.05 dla poziomu ufności 95%).

    Returns:
        tuple[float, float]: Krotka zawierająca dolną i górną granicę przedziału ufności dla proporcji.

    Example:
        >>> esty_prop(["niebieski", "zielony", "czerwony", "żółty", "czerwony", "czerwony", 
        ...            "zielony", "zielony", "żółty", "czerwony", "zielony", "żółty", "niebieski"], "czerwony", 0.05)
        (0.056801752272596207, 0.5585828631120192)
    """
    n = len(lista)
    p_hat = lista.count(wart) / n
    r = norm.ppf(1 - alfa / 2) * math.sqrt(p_hat * (1 - p_hat) / n)
    return float(p_hat - r), float(p_hat + r)

"""
Zestaw narzędzi do testowania podstawowych hipotez statystycznych.
Zawiera testy parametryczne dla średnich, wariancji oraz testy zgodności.
Wszystkie funkcje zwracają wartość True, gdy nie ma podstaw do odrzucenia
hipotezy zerowej (H0), lub False, gdy należy ją odrzucić na rzecz hipotezy alternatywnej.
"""


def hipoteza_sredniej_jedna_proba(
    probka: list[float],
    srednia_h0: float,
    poziom_istotnosci: float,
    odchylenie_populacji: Optional[float] = None,
    typ_hipotezy: Literal["L", "P", "O"] = "O"
) -> bool:
    """
    Weryfikuje hipotezę o wartości średniej populacji na podstawie jednej próby.
    
    W zależności od tego, czy znamy odchylenie standardowe populacji oraz 
    od wielkości próby, funkcja automatycznie dobiera odpowiedni test: 
    test Z (dla znanego odchylenia lub n > 30) lub test t-Studenta (dla n <= 30).
    
    Parameters:
        probka (list[float]): Pomiary w badanej próbie.
        srednia_h0 (float): Zakładana wartość średnia populacji (hipoteza H0).
        poziom_istotnosci (float): Dopuszczalne ryzyko błędu I rodzaju (alfa), np. 0.05.
        odchylenie_populacji (float, opcjonalnie): Znane odchylenie populacji. 
            Jeśli brak, funkcja wyestymuje je na podstawie próby.
        typ_hipotezy (Literal["L", "P", "O"]): Rodzaj testu: Lewostronny, Prawostronny lub Obustronny.
        
    Returns:
        bool: True, jeśli utrzymujemy hipotezę zerową (H0); False, jeśli ją odrzucamy.
        
    Example:
        >>> probka_makaronu = [460, 462, 458, 466, 457]
        >>> test_sredniej_jedna_proba(probka_makaronu, 454.0, 0.01, typ_hipotezy="O")
        False  # Odrzucamy H0, faktyczna średnia jest inna.
    """
    n = len(probka)
    srednia_proby = sum(probka) / n

    if odchylenie_populacji is not None:
        sigma = odchylenie_populacji
        uzyj_rozkadu_normalnego = True
    else:
        sigma = math.sqrt(no_estymator_wariancji(probka)) if n > 1 else 0.0
        uzyj_rozkadu_normalnego = n > 30

    statystyka = (srednia_proby - srednia_h0) * math.sqrt(n) / sigma

    if uzyj_rozkadu_normalnego:
        if typ_hipotezy == 'L':
            war_kryt = norm.ppf(poziom_istotnosci)
            return bool(statystyka > war_kryt)
        elif typ_hipotezy == 'P':
            war_kryt = norm.ppf(1 - poziom_istotnosci)
            return bool(statystyka < war_kryt)
        else: # Obustronny
            war_kryt = norm.ppf(1 - poziom_istotnosci / 2)
            return bool(-war_kryt < statystyka < war_kryt)
    else:
        stopien_swobody = n - 1
        if typ_hipotezy == 'L':
            war_kryt = t.ppf(poziom_istotnosci, stopien_swobody)
            return bool(statystyka > war_kryt)
        elif typ_hipotezy == 'P':
            war_kryt = t.ppf(1 - poziom_istotnosci, stopien_swobody)
            return bool(statystyka < war_kryt)
        else: # Obustronny
            war_kryt = t.ppf(1 - poziom_istotnosci / 2, stopien_swobody)
            return bool(-war_kryt < statystyka < war_kryt)


def hipoteza_porownania_srednich(
    probka1: list[float],
    probka2: list[float],
    poziom_istotnosci: float,
    odchylenie_pop_1: Optional[float] = None,
    odchylenie_pop_2: Optional[float] = None,
    typ_hipotezy: Literal["L", "P", "O"] = "O"
) -> bool:
    """
    Testuje hipotezę o równości dwóch średnich niezależnych próbek.
    
    Automatycznie dobiera test Z (jeśli znane są odchylenia obu populacji)
    lub test t-Studenta dla dwóch prób (stosując model ze spulowaną wariancją).
    
    Parameters:
        probka1 (list[float]): Pomiary z pierwszej grupy.
        probka2 (list[float]): Pomiary z drugiej grupy.
        poziom_istotnosci (float): Poziom istotności testu (alfa).
        odchylenie_pop_1 (float, opcjonalnie): Odchylenie standardowe populacji pierwszej.
        odchylenie_pop_2 (float, opcjonalnie): Odchylenie standardowe populacji drugiej.
        typ_hipotezy (Literal["L", "P", "O"]): Lewostronny, Prawostronny, lub Obustronny.
        
    Returns:
        bool: True, jeśli utrzymujemy hipotezę zerową (brak różnic w średnich).
        
    Example:
        >>> test_porownania_srednich([2.1, 5.3, 1.4], [1.9, 0.5, 2.8], 0.1, typ_hipotezy="P")
        True
    """
    n1, n2 = len(probka1), len(probka2)
    srednia1, srednia2 = sum(probka1) / n1, sum(probka2) / n2

    if odchylenie_pop_1 is not None and odchylenie_pop_2 is not None:
        blad_standardowy = math.sqrt((odchylenie_pop_1**2 / n1) + (odchylenie_pop_2**2 / n2))
        statystyka = (srednia1 - srednia2) / blad_standardowy

        if typ_hipotezy == 'L':
            wart_kryt = norm.ppf(poziom_istotnosci)
            return bool(statystyka > wart_kryt)
        elif typ_hipotezy == 'P':
            wart_kryt = norm.ppf(1 - poziom_istotnosci)
            return bool(statystyka < wart_kryt)
        else: # Obustronny
            wart_kryt = norm.ppf(1 - poziom_istotnosci / 2)
            return bool(-wart_kryt < statystyka < wart_kryt)
    else:
        war1, war2 = no_estymator_wariancji(probka1), no_estymator_wariancji(probka2)
        wariancja_spulowana = ((n1 - 1) * war1 + (n2 - 1) * war2) / (n1 + n2 - 2)
        blad_standardowy = math.sqrt(wariancja_spulowana * (1 / n1 + 1 / n2))
        
        statystyka = (srednia1 - srednia2) / blad_standardowy
        df = n1 + n2 - 2

        if typ_hipotezy == 'L':
            wart_kryt = t.ppf(poziom_istotnosci, df)
            return bool(statystyka > wart_kryt)
        elif typ_hipotezy == 'P':
            wart_kryt = t.ppf(1 - poziom_istotnosci, df)
            return bool(statystyka < wart_kryt)
        else: # Obustronny
            wart_kryt = t.ppf(1 - poziom_istotnosci / 2, df)
            return bool(-wart_kryt < statystyka < wart_kryt)


def hipoteza_zmienne_zalezne(
    probka1: list[float],
    probka2: list[float],
    roznica_h0: float,
    poziom_istotnosci: float,
    typ_hipotezy: Literal["L", "P", "O"] = "O"
) -> bool:
    """
    Weryfikuje hipotezę o braku różnic w powiązanych pomiarach 
    (np. badanie tego samego pacjenta przed i po przyjęciu leku).

    Parameters:
        probka2 (list[float]): Pomiary 1
        probka1 (list[float]): Pomiary 2
        roznica_h0 (float): Hipotetyczna średnia różnica (H0), przeważnie 0.
        poziom_istotnosci (float): Poziom istotności (alfa).
        typ_hipotezy (Literal["L", "P", "O"]): Kierunek hipotezy.
        
    Returns:
        bool: True, jeśli utrzymujemy hipotezę o braku zmian.
        
    Example:
        >>> przed = [13, 12, 16, 9]
        >>> po = [17, 11, 22, 18]
        >>> test_zmiennych_zaleznych(przed, po, 0, 0.05, typ_hipotezy="O")
        False
    """
    if len(probka1) != len(probka2):
        raise ValueError("Próbki zależne muszą być tej samej długości!")

    roznice = [a - b for a, b in zip(probka1, probka2)]
    n = len(roznice)
    d_srednie = sum(roznice) / n

    s_d = no_estymator_wariancji(roznice)

    statystyka = (d_srednie - roznica_h0) * math.sqrt(n) / s_d
    match typ_hipotezy:
        case 'L':
            wart_kryt = -t.ppf(1 - poziom_istotnosci, n - 1)
            return bool(statystyka > wart_kryt)
        case 'R':
            wart_kryt = t.ppf(1 - poziom_istotnosci, n - 1)
            return bool(statystyka < wart_kryt)  
        case _:
            wart_kryt = t.ppf(1 - poziom_istotnosci / 2, n - 1)
            return bool(-wart_kryt < statystyka < wart_kryt)


def hipoteza_wariancji(
    probka: list[float],
    wariancja_h0: float,
    poziom_istotnosci: float,
    typ_testu: Literal["L", "P", "O"] = "O"
) -> bool:
    """
    Weryfikuje hipotezę o określonej wartości wariancji populacji 
    w oparciu o rozkład chi-kwadrat.
    
    Parameters:
        probka (list[float]): Pomiary z próby.
        wariancja_h0 (float): Hipotetyczna wariancja populacji.
        poziom_istotnosci (float): Poziom istotności testu (alfa).
        typ_testu (Literal["L", "P", "O"]): Lewostronny, Prawostronny lub Obustronny.
        
    Returns:
        bool: True, jeśli utrzymujemy hipotezę zerową (wariancja populacji wynosi wariancja_h0).
        
    Example:
        >>> test_wariancji([1.2, 1.1, 1.4, 0.9, 1.0], 0.1, 0.05, typ_testu="O")
        True
    """
    n = len(probka)
    wariancja_proby = no_estymator_wariancji(probka)
    statystyka = (n - 1) * wariancja_proby / wariancja_h0
    df = n - 1

    if typ_testu == 'L':
        wart_kryt = chi2.ppf(poziom_istotnosci, df)
        return bool(statystyka > wart_kryt)
    elif typ_testu == 'P':
        wart_kryt = chi2.ppf(1 - poziom_istotnosci, df)
        return bool(statystyka < wart_kryt)
    else: # Obustronny
        wart_kryt_dolna = chi2.ppf(poziom_istotnosci / 2, df)
        wart_kryt_gorna = chi2.ppf(1 - poziom_istotnosci / 2, df)
        return bool(wart_kryt_dolna < statystyka < wart_kryt_gorna)


def hipoteza_zgodnosci_rownomierny(obserwacje: list[int], poziom_istotnosci: float) -> bool:
    """
    Przeprowadza test zgodności chi2, sprawdzając, czy dane 
    zostały pobrane z równomiernego rozkładu prawdopodobieństwa.
    
    Zakłada, że każda kategoria (koszyk) w idealnym świecie miałaby
    dokładnie taką samą oczekiwaną liczbę wystąpień (równą średniej).
    Testy zgodności klasycznie przeprowadzane są wyłącznie jako testy prawostronne.
    
    Parameters:
        obserwacje (list[int]): lista zliczeń zdarzeń w poszczególnych "koszykach".
        poziom_istotnosci (float): Poziom istotności testu (alfa).
        
    Returns:
        bool: True, jeśli dane zachowują się jak pochodzące z rozkładu równomiernego.
        
    Example:
        >>> test_zgodnosci_rownomierny([20, 30, 40], 0.05)
        False  # Dane zbyt mocno odchylają się od oczekiwanej wartości [30, 30, 30].
    """
    liczba_kategorii = len(obserwacje)
    oczekiwana = sum(obserwacje) / liczba_kategorii
    statystyka = sum(((obs - oczekiwana) ** 2) / oczekiwana for obs in obserwacje)
    
    df = liczba_kategorii - 1
    war_kryt = chi2.ppf(1 - poziom_istotnosci, df)
    
    return bool(statystyka < war_kryt)


def hipoteza_zgodnosci_poisson(
    obserwacje: list[int],
    lam: float,
    poziom_istotnosci: float
) -> bool:
    """
    Przeprowadza test zgodności chi2 w celu weryfikacji, czy liczebności
    danych kategorii odpowiadają teoretycznemu rozkładowi Poissona z parametrem lambda.
    
    Uwaga: Funkcja zakłada, że element listy pod indeksem 'i' reprezentuje liczbę
    wystąpień zdarzenia zliczonego jako równo 'i' (np. lista[0] to liczba dni 
    z 0 awariami, lista[1] to liczba dni z 1 awarią, itd.).
    
    Parameters:
        obserwacje (list[int]): lista zliczeń (indeks odpowiada liczbie badanych rzadkich zdarzeń).
        lambda_param (float): Hipotetyczny parametr intensywności lambda dla rozkładu Poissona.
        poziom_istotnosci (float): Poziom istotności testu (alfa).
        
    Returns:
        bool: True, jeśli można założyć pochodzenie danych z rozkładu Poissona.
        
    Example:
        >>> awarie_wodociagowe = [10, 27, 29, 16, 8, 7]
        >>> test_zgodnosci_poissona(awarie_wodociagowe, lambda_param=2.0, poziom_istotnosci=0.05)
        True
    """
    n = sum(obserwacje)
    liczba_kategorii = len(obserwacje)
    
    oczekiwane = [
        n * ((lam ** k) * math.exp(-lam) / math.factorial(k))
        for k in range(liczba_kategorii)
    ]
    
    statystyka = sum(((obs - oczek) ** 2) / oczek for obs, oczek in zip(obserwacje, oczekiwane))
    df = liczba_kategorii - 1
    
    war_kryt = chi2.ppf(1 - poziom_istotnosci, df)
    
    return bool(statystyka < war_kryt)