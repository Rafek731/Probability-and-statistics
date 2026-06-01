import math
from typing import Any
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

def no_estymator_wariacji(dane: list[float]) -> float:
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
    sq_diff = [(x - x_bar)**2 for x in dane]
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
    s = math.sqrt(no_estymator_wariacji(dane))

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
    numerator = (n - 1) * no_estymator_wariacji(dane)
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