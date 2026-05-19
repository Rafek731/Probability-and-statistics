import math
from typing import List, Tuple
from scipy.stats import norm

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


def poisson_ext(n: int, p: float, k: List[int]) -> Tuple[float, float]:
    """
    Szacuje prawdopodobieństwo uzyskania liczby sukcesów ze zbioru `k` oraz błąd oszacowania.
    
    Wykorzystuje przybliżenie Poissona dla sumy prawdopodobieństw z podanej listy
    oczekiwanych ilości sukcesów w schemacie Bernoulliego.

    Args:
        n (int): Liczba przeprowadzonych prób.
        p (float): Prawdopodobieństwo odniesienia sukcesu w pojedynczej próbie.
        k (List[int]): Lista interesujących nas ilości sukcesów (np. [0, 1, 2]).

    Returns:
        Tuple[float, float]: Krotka zawierająca:
            - Szacowane prawdopodobieństwo otrzymania liczby sukcesów podanej w liście.
            - Maksymalny błąd tego oszacowania.

    Example:
        >>> poisson_ext(10000000, 0.00000007, [0, 1, 2])
        (0.9658584158742916, 4.900000000000001e-08)
    """
    lam = n * p
    result = [poisson(n, p, k_int) for k_int in k]
    return sum(result), (lam**2) / n


def CTG(n: int, m: float, s: float, d: float = -math.inf, g: float = math.inf) -> float:
    """
    Szacuje prawdopodobieństwo na podstawie Centralnego Twierdzenia Granicznego (CTG).

    Oblicza prawdopodobieństwo tego, że suma `n` niezależnych zmiennych losowych
    o podanej wartości oczekiwanej i odchyleniu standardowym wpadnie w przedział [d, g].

    Args:
        n (int): Liczba niezależnych zmiennych losowych w sumie.
        m (float): Wartość oczekiwana (średnia) pojedynczej zmiennej losowej.
        s (float): Odchylenie standardowe pojedynczej zmiennej losowej.
        d (float, optional): Dolne ograniczenie przedziału. Domyślnie -nieskończoność.
        g (float, optional): Górne ograniczenie przedziału. Domyślnie +nieskończoność.

    Returns:
        float: Szacowane prawdopodobieństwo znalezienia się sumy w przedziale [d, g].

    Example:
        >>> CTG(400, 0.3, math.sqrt(0.3 * 0.7), d=130)
        0.13761676203741713
    """
    return norm.cdf((g - n * m) / (s * math.sqrt(n))) - norm.cdf((d - n * m) / (s * math.sqrt(n)))


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
    return math.ceil(d**2 / ((norm.ppf((2 - pb) / 2)**2) * p * (1 - p)))