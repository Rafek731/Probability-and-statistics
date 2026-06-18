import math
from typing import Any, Optional, Literal, Callable
from scipy.stats import norm, t, chi2


# ==========================================
#       CZĘŚĆ I: Twierdzenia Graniczne 
# ==========================================

def poisson_approx(n: int, p: float, k: int) -> float:
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
        >>> poisson_approx(10000000, 0.00000007, 1)
        0.34760971265398666
    """
    lam = n * p
    return poisson(lam, k)

def poisson(lam: float, k : int) -> float:
    """
    Oblicza prawdopodobieństwo `k` sukcesów w rozkładzie Poissona z parametrem `lam`.

    Args:
        lam (float): Parametr lambda rozkładu Poissona.
        k (int): Ilość sukcesów.

    Returns:
        float: Prawdopodobieństwo otrzymania dokładnie `k` sukcesów.
    """
    result = math.exp(-lam)
    for i in range(1, k + 1):
        result *= lam / i
    return result

def multi_poisson_z_bledem(n: int, p: float, ks: list[int]) -> tuple[float, float]:
    """
    Szacuje prawdopodobieństwo uzyskania liczby sukcesów ze zbioru `ks` oraz błąd oszacowania.
    
    Wykorzystuje przybliżenie Poissona dla sumy prawdopodobieństw z podanej listy
    oczekiwanych ilości sukcesów w schemacie Bernoulliego.

    Args:
        n (int): Liczba przeprowadzonych prób.
        p (float): Prawdopodobieństwo odniesienia sukcesu w pojedynczej próbie.
        ks (list[int]): Lista interesujących nas ilości sukcesów (np. [0, 1, 2]).

    Returns:
        tuple[float, float]: Krotka zawierająca:
            - Szacowane prawdopodobieństwo otrzymania liczby sukcesów podanej w liście.
            - Maksymalny błąd tego oszacowania.

    Example:
        >>> multi_poisson_z_bledem(10000000, 0.00000007, [0, 1, 2])
        (0.9658584158742916, 4.900000000000001e-08)
    """
    lam = n * p
    result = 0
    for k in ks:
        result += poisson(lam, k)
    return result, (lam ** 2) / n


def CTG(n: int, mu: float, sigma: float, dol: float = -math.inf, gora: float = math.inf) -> float:
    """
    Szacuje prawdopodobieństwo na podstawie Centralnego Twierdzenia Granicznego (CTG).

    Oblicza prawdopodobieństwo tego, że suma `n` niezależnych zmiennych losowych
    o podanej wartości oczekiwanej i odchyleniu standardowym wpadnie w przedział [dol, gora].
    Korzysta ze wzoru standaryzującego: (S_n - n * mu) / (sigma * sqrt(n)).

    Args:
        n (int): Liczba niezależnych zmiennych losowych w sumie.
        mu (float): Wartość oczekiwana (średnia) pojedynczej zmiennej losowej.
        sigma (float): Odchylenie standardowe pojedynczej zmiennej losowej.
        dol (float, optional): Dolne ograniczenie przedziału. Domyślnie -inf.
        gora (float, optional): Górne ograniczenie przedziału. Domyślnie +inf.

    Returns:
        float: Szacowane prawdopodobieństwo znalezienia się sumy w przedziale [dol, gora].

    Example:
        >>> CTG(400, 0.3, math.sqrt(0.3 * 0.7), dol=130)
        0.13761676203741713
    """
    return norm.cdf((gora - n * mu) / (sigma * math.sqrt(n))) - norm.cdf((dol - n * mu) / (sigma * math.sqrt(n)))


def rozmiar(p: float, d: float, pb: float) -> int:
    """
    Oblicza minimalny rozmiar próby wymagany do osiągnięcia zadanego błędu oszacowania
    dla SUMY sukcesów w schemacie Bernoulliego (S_n).

    Args:
        p (float): Prawdopodobieństwo uzyskania sukcesu w pojedynczej próbie.
        d (float): Rozmiar dopuszczalnego odchylenia (dla absolutnej liczby sukcesów, nie dla proporcji).
        pb (float): Poziom istotności (alfa), czyli prawdopodobieństwo popełnienia błędu.

    Returns:
        int: Minimalny rozmiar próby (zaokrąglony w górę do pełnej liczby całkowitej).

    Example:
        >>> rozmiar(0.5, 3.375, 0.5)
        101
    """
    return math.ceil((d ** 2) / ((norm.ppf(1 - pb/2) ** 2) * p * (1 - p)))


# ==========================================
#           CZĘŚĆ II: Estymatory 
# ==========================================

def no_estymator_wariancji(dane: list[float]) -> float:
    """
    Oblicza nieobciążony estymator wariancji dla podanej próby.

    Args:
        dane (list[float]): lista zawierająca wyniki pobrane z próby.

    Returns:
        float: Wartość nieobciążonego estymatora wariancji.

    Example:
        >>> no_estymator_wariancji([1, 2, 3, 4, 5, 6, 7, 8, 9])
        7.5
    """
    n = len(dane)
    x_bar = sum(dane) / n
    sq_diff = list(map(lambda x: (x - x_bar)**2, dane))
    return sum(sq_diff) / (n - 1)


def przedzial_ufnosci_srednia(dane: list[float], sigma: float, alfa: float) -> tuple[float, float]:
    """
    Wyznacza przedział ufności dla wartości średniej przy znanym odchyleniu standardowym populacji.

    Args:
        dane (list[float]): lista zawierająca wyniki pobrane z próby.
        sigma (float): Znane odchylenie standardowe w populacji.
        alfa (float): Poziom błędu (np. 0.05 dla poziomu ufności 95%).

    Returns:
        tuple[float, float]: Krotka zawierająca dolną i górną granicę przedziału ufności.

    Example:
        >>> przedzial_ufnosci_srednia([2, 2.3, 2.4, 2.5, 1.9, 2.3, 2.5, 2.1, 2.4, 2.3], 1, 0.05)
        (1.6502049676954385, 2.8897950323045616)
    """
    n = len(dane)
    x_bar = sum(dane) / n
    r = norm.ppf(1 - alfa / 2) * sigma / math.sqrt(n)
    return float(x_bar - r), float(x_bar + r)


def przedzial_ufnosci_srednia_bez_od(dane: list[float], alfa: float) -> tuple[float, float]:
    """
    Wyznacza przedział ufności dla wartości średniej przy nieznanym odchyleniu standardowym populacji.
    
    Automatycznie dobiera odpowiedni rozkład w zależności od rozmiaru próby:
    - Dla małych prób (n <= 30) korzysta z rozkładu t-Studenta.
    - Dla dużych prób (n > 30) korzysta ze standardowego rozkładu normalnego.

    Args:
        dane (list[float]): lista zawierająca wyniki pobrane z próby.
        alfa (float): Poziom błędu (np. 0.05 dla poziomu ufności 95%).

    Returns:
        tuple[float, float]: Krotka zawierająca dolną i górną granicę przedziału ufności.

    Example:
        >>> przedzial_ufnosci_srednia_bez_od([2, 2.3, 2.4, 2.5, 1.9, 2.3, 2.5, 2.1, 2.4, 2.3], 0.05)
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
        tuple[float, float]: Krotka zawierająca dolną i górną granicę przedziału ufności dla wariancji.

    Example:
        >>> przedzial_ufnosci_wariancja([2, 2.3, 2.4, 2.5, 1.9, 2.3, 2.5, 2.1, 2.4, 2.3], 0.05)
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
        >>> przedzial_ufnosci_proporcja(["niebieski", "zielony", "czerwony", "żółty", "czerwony", "czerwony", 
        ...            "zielony", "zielony", "żółty", "czerwony", "zielony", "żółty", "niebieski"], "czerwony", 0.05)
        (0.056801752272596207, 0.5585828631120192)
    """
    n = len(lista)
    p_hat = lista.count(wart) / n
    r = norm.ppf(1 - alfa / 2) * math.sqrt(p_hat * (1 - p_hat) / n)
    return float(p_hat - r), float(p_hat + r)

# ==========================================
#          CZĘŚĆ III: Hipotezy 
# ==========================================


def hipoteza_srednia(
    probka: list[float],
    srednia_h0: float,
    alfa: float,
    sigma: Optional[float] = None,
    typ_hipotezy: Literal["L", "P", "O"] = "O"
) -> bool:
    """
    Weryfikuje hipotezę o wartości średniej populacji na podstawie jednej próby.
    W skrócie - mówi, czy utrzymujemy hipotezę zerową.
    
    W zależności od tego, czy znamy odchylenie standardowe populacji oraz 
    od wielkości próby, funkcja automatycznie dobiera odpowiedni test: 
    test Z (dla znanego odchylenia lub n > 30) lub test t-Studenta (dla n <= 30).
    
    Parameters:
        probka (list[float]): Pomiary w badanej próbie.
        srednia_h0 (float): Zakładana wartość średnia populacji (hipoteza H0).
        alfa (float): Dopuszczalne ryzyko błędu I rodzaju, np. 0.05.
        sigma (float, opcjonalnie): Znane odchylenie populacji. 
            Jeśli brak, funkcja wyestymuje je na podstawie próby.
        typ_hipotezy (Literal["L", "P", "O"]): Rodzaj hipotezy: Lewostronna, Prawostronna lub Obustronna.
        
    Returns:
        bool: True, jeśli utrzymujemy hipotezę zerową (H0); False, jeśli ją odrzucamy.
        
    Example:
        >>> probka_makaronu = [460, 462, 458, 466, 457]
        >>> hipoteza_srednia(probka_makaronu, 454.0, 0.01, typ_hipotezy="O")
        False  # Odrzucamy H0, faktyczna średnia jest inna.
    """
    n = len(probka)
    srednia_proby = sum(probka) / n

    if sigma is not None:
        uzyj_rozkadu_normalnego = True
    else:
        sigma = math.sqrt(no_estymator_wariancji(probka)) if n > 1 else 0.0
        uzyj_rozkadu_normalnego = n > 30

    statystyka = (srednia_proby - srednia_h0) * math.sqrt(n) / sigma

    if uzyj_rozkadu_normalnego:
        if typ_hipotezy == 'L':
            wart_kryt = norm.ppf(alfa)
            return bool(statystyka > wart_kryt)
        elif typ_hipotezy == 'P':
            wart_kryt = norm.ppf(1 - alfa)
            return bool(statystyka < wart_kryt)
        else: # Obustronna
            wart_kryt = norm.ppf(1 - alfa / 2)
            return bool(-wart_kryt < statystyka < wart_kryt)
    else:
        df = n - 1
        if typ_hipotezy == 'L':
            wart_kryt = -t.ppf(1 - alfa, df)
            return bool(statystyka > wart_kryt)
        elif typ_hipotezy == 'P':
            wart_kryt = t.ppf(1 - alfa, df)
            return bool(statystyka < wart_kryt)
        else: # Obustronna
            wart_kryt = t.ppf(1 - alfa / 2, df)
            return bool(-wart_kryt < statystyka < wart_kryt)


def hipoteza_porownanie_srednich(
    probka1: list[float],
    probka2: list[float],
    alfa: float,
    sigma1: Optional[float] = None,
    sigma2: Optional[float] = None,
    typ_hipotezy: Literal["L", "P", "O"] = "O"
) -> bool:
    """
    Testuje hipotezę o równości dwóch średnich niezależnych próbek.
    
    Automatycznie dobiera test Z (jeśli znane są odchylenia obu populacji)
    lub test t-Studenta dla dwóch prób.
    
    Parameters:
        probka1 (list[float]): Pomiary z pierwszej grupy.
        probka2 (list[float]): Pomiary z drugiej grupy.
        alfa (float): Poziom istotności testu (alfa).
        sigma1 (float, opcjonalnie): Odchylenie standardowe populacji pierwszej.
        sigma2 (float, opcjonalnie): Odchylenie standardowe populacji drugiej.
        typ_hipotezy (Literal["L", "P", "O"]): Lewostronny, Prawostronny, lub Obustronny.
        
    Returns:
        bool: True, jeśli utrzymujemy hipotezę zerową (brak różnic w średnich).
        
    Example:
        >>> hipoteza_porownanie_srednich([2.1, 5.3, 1.4], [1.9, 0.5, 2.8], 0.1, typ_hipotezy="P")
        True
    """
    n1, n2 = len(probka1), len(probka2)
    srednia1, srednia2 = sum(probka1) / n1, sum(probka2) / n2

    if sigma1 is not None and sigma2 is not None:
        mianownik = math.sqrt((sigma1**2 / n1) + (sigma2**2 / n2))
        statystyka = (srednia1 - srednia2) / mianownik

        if typ_hipotezy == 'L':
            wart_kryt = -norm.ppf(1 - alfa)
            return bool(statystyka > wart_kryt)
        elif typ_hipotezy == 'P':
            wart_kryt = norm.ppf(1 - alfa)
            return bool(statystyka < wart_kryt)
        else: # Obustronny
            wart_kryt = norm.ppf(1 - alfa / 2)
            return bool(-wart_kryt < statystyka < wart_kryt)
    else:
        war1, war2 = no_estymator_wariancji(probka1), no_estymator_wariancji(probka2)
        wariancja_spulowana = ((n1 - 1) * war1 + (n2 - 1) * war2) / (n1 + n2 - 2)
        mianownik = math.sqrt(wariancja_spulowana * (1 / n1 + 1 / n2))
        
        statystyka = (srednia1 - srednia2) / mianownik
        df = n1 + n2 - 2

        if typ_hipotezy == 'L':
            wart_kryt = -t.ppf(1 - alfa, df)
            return bool(statystyka > wart_kryt)
        elif typ_hipotezy == 'P':
            wart_kryt = t.ppf(1 - alfa, df)
            return bool(statystyka < wart_kryt)
        else: # Obustronny
            wart_kryt = t.ppf(1 - alfa / 2, df)
            return bool(-wart_kryt < statystyka < wart_kryt)


def hipoteza_zmienne_zalezne(
    probka1: list[float],
    probka2: list[float],
    roznica_h0: float,
    alfa: float,
    typ_hipotezy: Literal["L", "P", "O"] = "O"
) -> bool:
    """
    Weryfikuje hipotezę o braku różnic w powiązanych pomiarach 
    (np. badanie tego samego pacjenta przed i po przyjęciu leku).

    Parameters:
        probka1 (list[float]): Pomiary 1
        probka2 (list[float]): Pomiary 2
        roznica_h0 (float): Hipotetyczna średnia różnica (H0), przeważnie 0.
        alfa (float): Poziom istotności.
        typ_hipotezy (Literal["L", "P", "O"]): Kierunek hipotezy.
        
    Returns:
        bool: True, jeśli utrzymujemy hipotezę o braku zmian.
        
    Example:
        >>> przed = [13, 12, 16, 9]
        >>> po = [17, 11, 22, 18]
        >>> hipoteza_zmienne_zalezne(przed, po, 0, 0.05, typ_hipotezy="O")
        False
    """
    if len(probka1) != len(probka2):
        raise ValueError("Próbki zależne muszą być tej samej długości!")

    roznice = [a - b for a, b in zip(probka1, probka2)]
    n = len(roznice)
    d_srednie = sum(roznice) / n

    s_d = math.sqrt(no_estymator_wariancji(roznice))

    statystyka = (d_srednie - roznica_h0) * math.sqrt(n) / s_d
    if typ_hipotezy == 'L':
        wart_kryt = -t.ppf(1 - alfa, n - 1)
        return bool(statystyka > wart_kryt)
    elif typ_hipotezy == 'P':
        wart_kryt = t.ppf(1 - alfa, n - 1)
        return bool(statystyka < wart_kryt)  
    else:
        wart_kryt = t.ppf(1 - alfa / 2, n - 1)
        return bool(-wart_kryt < statystyka < wart_kryt)


def hipoteza_wariancja(
    probka: list[float],
    wariancja_h0: float,
    alfa: float,
    typ_testu: Literal["L", "P", "O"] = "O"
) -> bool:
    """
    Weryfikuje hipotezę o określonej wartości wariancji populacji 
    w oparciu o rozkład chi-kwadrat.
    
    Parameters:
        probka (list[float]): Pomiary z próby.
        wariancja_h0 (float): Hipotetyczna wariancja populacji.
        alfa (float): Poziom istotności testu.
        typ_testu (Literal["L", "P", "O"]): Lewostronny, Prawostronny lub Obustronny.
        
    Returns:
        bool: True, jeśli utrzymujemy hipotezę zerową (wariancja populacji wynosi wariancja_h0).
        
    Example:
        >>> hipoteza_wariancja([1.2, 1.1, 1.4, 0.9, 1.0], 0.1, 0.05, typ_testu="O")
        True
    """
    n = len(probka)
    wariancja_proby = no_estymator_wariancji(probka)
    statystyka = (n - 1) * wariancja_proby / wariancja_h0
    df = n - 1

    if typ_testu == 'L':
        wart_kryt = chi2.ppf(alfa, df)
        return bool(statystyka > wart_kryt)
    elif typ_testu == 'P':
        wart_kryt = chi2.ppf(1 - alfa, df)
        return bool(statystyka < wart_kryt)
    else: # Obustronny
        wart_kryt_dolna = chi2.ppf(alfa / 2, df)
        wart_kryt_gorna = chi2.ppf(1 - alfa / 2, df)
        return bool(wart_kryt_dolna < statystyka < wart_kryt_gorna)


def spr_rowno(obserwacje: list[int], alfa: float) -> bool:
    """
    Przeprowadza test zgodności chi2, sprawdzając, czy dane 
    zostały pobrane z równomiernego rozkładu prawdopodobieństwa.
    
    Zakłada, że każda kategoria (koszyk) w idealnym świecie miałaby
    dokładnie taką samą oczekiwaną liczbę wystąpień (równą średniej).
    Testy zgodności klasycznie przeprowadzane są wyłącznie jako testy prawostronne.
    
    Parameters:
        obserwacje (list[int]): lista zliczeń zdarzeń w poszczególnych "koszykach".
        alfa (float): Poziom istotności testu.
        
    Returns:
        bool: True, jeśli dane zachowują się jak pochodzące z rozkładu równomiernego.
        
    Example:
        >>> spr_rowno([20, 30, 40], 0.05)
        False  # Dane zbyt mocno odchylają się od oczekiwanej wartości [30, 30, 30].
    """
    n = len(obserwacje)
    oczekiwane = sum(obserwacje) / n
    statystyka = sum(((obs - oczekiwane) ** 2) / oczekiwane for obs in obserwacje)
    
    df = n - 1
    wart_kryt = chi2.ppf(1 - alfa, df)
    
    return bool(statystyka < wart_kryt)


def spr_poisson(
    obserwacje: list[int],
    lam: float,
    alfa: float
) -> bool:
    """
    Przeprowadza test zgodności chi2 w celu weryfikacji, czy liczebności
    danych kategorii odpowiadają teoretycznemu rozkładowi Poissona z parametrem lambda.
    
    Uwaga: Funkcja zakłada, że element listy pod indeksem 'i' reprezentuje liczbę
    wystąpień zdarzenia zliczonego jako równo 'i' (np. lista[0] to liczba dni 
    z 0 awariami, lista[1] to liczba dni z 1 awarią, itd.).
    
    Parameters:
        obserwacje (list[int]): lista zliczeń (indeks odpowiada liczbie badanych rzadkich zdarzeń).
        lam (float): Hipotetyczny parametr intensywności lambda dla rozkładu Poissona.
        alfa (float): Poziom istotności testu.
        
    Returns:
        bool: True, jeśli można założyć pochodzenie danych z rozkładu Poissona.
        
    Example:
        >>> awarie_wodociagowe = [10, 27, 29, 16, 8, 7]
        >>> spr_poisson(awarie_wodociagowe, lam=2.0, alfa=0.05)
        True
    """
    suma = sum(obserwacje)
    n = len(obserwacje)
    # Liczymy wszystkie koszyki oprócz ostatniego
    oczekiwane = [
        suma * ((lam ** k) * math.exp(-lam) / math.factorial(k))
        for k in range(n - 1)
    ]
    # Ostatni koszyk zbiera całą resztę (ogon prawdopodobieństwa P(X >= n-1))
    oczekiwane.append(suma - sum(oczekiwane))
    
    statystyka = sum(((obs - oczek) ** 2) / oczek for obs, oczek in zip(obserwacje, oczekiwane))
    df = n - 2
    
    wart_kryt = chi2.ppf(1 - alfa, df)
    return bool(statystyka < wart_kryt)

# ==========================================
#       CZĘŚĆ IV: Metody monte carlo 
# ==========================================

def generator_jednostajny(n: int, seed: int = 42, a: float = 0.0, b: float = 1.0) -> list[float]:
    """
    Generuje ciąg liczb pseudolosowych z rozkładu jednostajnego na przedziale [a, b].

    Funkcja wykorzystuje prosty generator liniowy kongruentny oparty na algorytmie 
    Parka-Mullera. Wartości są skalowane do podanego przedziału.

    Args:
        n (int): Docelowa długość generowanego ciągu.
        seed (int, optional): Wartość początkowa (ziarno) algorytmu. Domyślnie 42.
        a (float, optional): Dolna granica przedziału. Domyślnie 0.0.
        b (float, optional): Górna granica przedziału. Domyślnie 1.0.

    Returns:
        list[float]: n-elementowa lista liczb pseudolosowych z przedziału [a, b].
    """
    if seed <= 0:
        raise ValueError("Ziarno (seed) w algorytmie Parka-Mullera musi być > 0.")
        
    A: int = 2147483647
    M: int = 16807
    
    stan: int = seed # Trzymamy stan jako czystą liczbę całkowitą
    ciag: list[float] = []
    
    for _ in range(n):
        stan = (stan * M) % A
        
        u = stan / A 
        ciag.append(a + u * abs(b - a))

    return ciag


def generator_wykladniczy(n: int, seed: int = 42, lam: float = 1.0) -> list[float]:
    """
    Generuje ciąg liczb pseudolosowych z rozkładu wykładniczego.

    Wykorzystuje metodę odwracania dystrybuanty na podstawie wygenerowanych 
    wcześniej liczb z rozkładu jednostajnego na przedziale [0, 1].

    Args:
        n (int): Docelowa długość generowanego ciągu.
        seed (int, optional): Wartość początkowa (ziarno) algorytmu. Domyślnie 42.
        lam (float, optional): Parametr lambda rozkładu wykładniczego. Domyślnie 1.0.

    Returns:
        list[float]: n-elementowa lista liczb pseudolosowych z rozkładu wykładniczego.
    """
    return [-math.log(1 - x) / lam for x in generator_jednostajny(n, seed, 0.0, 1.0)]


def generator_normalny(n: int, seed: int = 42) -> list[float]:
    """
    Generuje ciąg liczb pseudolosowych ze standardowego rozkładu normalnego.

    Funkcja implementuje transformację Boxa-Mullera z wykorzystaniem pary 
    niezależnych ciągów zmiennych losowych o rozkładzie jednostajnym na przedziale [0, 1].

    Args:
        n (int): Docelowa długość generowanego ciągu.
        seed (int, optional): Wartość początkowa (ziarno) algorytmu. Domyślnie 42.

    Returns:
        list[float]: n-elementowa lista liczb pseudolosowych z rozkładu normalnego 
        o wartości średniej 0 i wariancji 1.
    """
    ciag: list[float] = generator_jednostajny(2 * n, seed, 0.0, 1.0)
    xs: list[float] = ciag[:n]
    ys: list[float] = ciag[n:]
    gs: list[float] = []
    
    for x, y in zip(xs, ys):
        sq_ln: float = math.sqrt(-2 * math.log(x))
        gs.append(sq_ln * math.cos(2 * math.pi * y))
        
    return gs


def generator_kostka(n: int, k: int = 6, seed: int = 42) -> list[int]:
    """
    Generuje ciąg liczb pseudolosowych symulujących rzuty wielościenną kostką.

    Funkcja bazuje na rozkładzie jednostajnym na odcinku [1, k+1],
    z którego pobierana jest część całkowita z zachowaniem górnej granicy.

    Args:
        n (int): Docelowa długość generowanego ciągu (liczba rzutów).
        k (int, optional): Liczba ścianek na kostce. Domyślnie 6.
        seed (int, optional): Wartość początkowa (ziarno) algorytmu. Domyślnie 42.

    Returns:
        list[int]: n-elementowa lista wyników rzutu k-ścienną kostką.
    """
    return [min(math.floor(x), k) for x in generator_jednostajny(n, seed, 1.0, k + 1.0)]


def calka_MC(
    fun: Callable[[float], float], 
    n: int = 1000000, 
    a: float = 0.0, 
    b: float = 1.0, 
    seed: int = 42
) -> float:
    """
    Szacuje wartość całki oznaczonej z funkcji na przedziale metodą Monte Carlo.

    Oblicza wartość oczekiwaną dla próbek funkcji na losowych punktach, 
    korzystając z wygenerowanego ciągu o rozkładzie jednostajnym.

    Args:
        fun (Callable[[float], float]): Funkcja przyjmująca argument typu float 
            i zwracająca wartość typu float, której całkę chcemy oszacować.
        n (int, optional): Długość ciągu liczb pseudolosowych użytych do szacowania. 
            Domyślnie 1 000 000.
        a (float, optional): Dolna granica całkowania. Domyślnie 0.0.
        b (float, optional): Górna granica całkowania. Domyślnie 1.0.
        seed (int, optional): Wartość początkowa (ziarno) algorytmu. Domyślnie 42.

    Returns:
        float: Przybliżona wartość całki oznaczonej funkcji na przedziale [a, b].
    """
    return abs(b - a) * sum([fun(x) for x in generator_jednostajny(n, seed, a, b)]) / n