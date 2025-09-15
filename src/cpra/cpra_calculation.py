"""
CPRA (Calculated Panel Reactive Antibody) Calculation Module

This module provides functions to calculate CPRA for organ transplant candidates
using two different methods: donor filtering and haplotype frequency.
"""
# %%

import numpy as np
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def _cpra_donor_filtering_single(
    uas: list[str],
    donor_hla: pd.DataFrame,
    donor_hla_columns: dict[str, str] = {"donor_id": "id", "hla": "hla"},
) -> float:
    """
    Calculate CPRA (Calculated Panel Reactive Antibody) for a single recipient using
    the donor filtering method.

    Parameters
    ----------
    uas : list[str]
        List of unacceptable antigens for the recipient.
    donor_hla : pd.DataFrame
        Donor HLA data, including columns:
        - 'id': donor ID
        - 'hla': HLA allele

    Returns
    -------
    float
        CPRA percentage (0-100).

    Example
    -------
    >>> uas = ['DRB1*09:01', 'A*02:01']
    >>> donor_hla = pd.DataFrame({
    ...     'id': [1, 2, 2],
    ...     'hla': ['A*02:03', 'DRB1*09:01', 'B*07:02']
    ... })
    >>> cpra = _cpra_donor_filtering_single(uas, donor_hla)
    """

    # Standardize donor column names
    donor_hla = donor_hla.rename(
        columns={
            donor_hla_columns["donor_id"]: "id_d",
            donor_hla_columns["hla"]: "hla",
        }
    )[["id_d", "hla"]]

    # Filter donor_hla for rows where hla is in uas
    matched = donor_hla[donor_hla["hla"].isin(uas)]

    # Count unique donor IDs with at least one unacceptable antigen
    n_matched = matched["id_d"].nunique()

    # Total number of unique donors
    n_total = donor_hla["id_d"].nunique()

    # Calculate CPRA percentage
    cpra = (n_matched / n_total) * 100 if n_total > 0 else 0.0

    return cpra


def cpra_donor_filtering(
    unacceptable_antigen: pd.DataFrame,
    donor_hla: pd.DataFrame,
    unacceptable_antigen_columns: dict[str, str] = {
        "recipient_id": "id",
        "unacceptable_antigen": "uas",
    },
    donor_hla_columns: dict[str, str] = {"donor_id": "id", "hla": "hla"},
) -> pd.DataFrame:
    """
    Calculate CPRA (Calculated Panel Reactive Antibody) for each recipient using
    donor filtering method.

    Parameters
    ----------
    unacceptable_antigen : pd.DataFrame
        Recipient unacceptable antigen data, including at least two columns:
        - recipient ID
        - unacceptable antigen (HLA allele)
    donor_hla : pd.DataFrame
        Donor HLA data, including at least two columns:
        - donor ID
        - HLA allele
    unacceptable_antigen_columns : dict[str, str]
        Column mapping for recipient unacceptable antigen data, with keys as standard
        names ("recipient_id", "unacceptable_antigen") and values as actual column
        names in `unacceptable_antigen`.
    donor_hla_columns : dict[str, str]
        Column mapping for donor HLA data, with keys as standard names ("donor_id",
        "hla) and values as actual column names in `donor_hla`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing recipient ID and corresponding CPRA percentage.

    Example
    -------
    >>> unacceptable_antigen = pd.DataFrame({
    ...     'id': [1, 1, 2],
    ...     'uas': ['DRB1*09:01', 'A*02:01', 'B*07:02']
    ... })
    >>> donor_hla = pd.DataFrame({
    ...     'id': [1, 2, 2],
    ...     'hla': ['A*02:03', 'DRB1*09:01', 'B*07:02']
    ... })
    >>> result = cpra_donor_filtering(unacceptable_antigen, donor_hla)
    """

    # Standardize donor and recipient column names
    donor_hla = donor_hla.rename(
        columns={
            donor_hla_columns["donor_id"]: "id_d",
            donor_hla_columns["hla"]: "hla",
        }
    )[["id_d", "hla"]]

    unacceptable_antigen = unacceptable_antigen.rename(
        columns={
            unacceptable_antigen_columns["recipient_id"]: "id_r",
            unacceptable_antigen_columns["unacceptable_antigen"]: "uas",
        }
    )[["id_r", "uas"]]

    # Merge unacceptable antigens and donor HLA
    merged = unacceptable_antigen.merge(
        donor_hla, left_on="uas", right_on="hla", how="left"
    )

    # Filter records with matched donors
    merged_filtered = merged.dropna(subset=["id_d"])

    # Count the number of matched donors for each recipient
    result = (
        merged_filtered.groupby("id_r")["id_d"]
        .nunique()
        .reset_index()
        .rename(columns={"id_d": "n"})
    )

    # Total number of donors
    n_donor = donor_hla["id_d"].nunique()

    # Calculate CPRA percentage
    result["cpra"] = (result["n"] / n_donor) * 100

    # Merge results, set CPRA to 0 for unmatched recipients
    cpra = (
        unacceptable_antigen[["id_r"]]
        .drop_duplicates()
        .merge(result[["id_r", "cpra"]], on="id_r", how="left")
    )
    cpra["cpra"] = cpra["cpra"].fillna(0)
    cpra = cpra.rename(columns={"id_r": unacceptable_antigen_columns["recipient_id"]})

    return cpra


def _cpra_haplotype_frequency_single(
    uas: list[str],
    haplotype_frequency: pd.DataFrame,
    loci: list[str] = ["A", "B", "C", "DRB1", "DQB1"],
    haplotype_frequency_column: dict[str, str] = {"frequency": "freq"},
) -> float:
    """
    Calculate CPRA (Calculated Panel Reactive Antibody) for a single recipient using
    the haplotype frequency method.

    This function uses the inclusion-exclusion principle to estimate the probability
    that a random donor has at least one unacceptable antigen for the recipient,
    based on population haplotype frequencies.

    Parameters
    ----------
    uas : list[str]
        List of unacceptable antigens for the recipient.
    haplotype_frequency : pd.DataFrame
        Haplotype frequency data, including columns for loci and haplotype frequency.
    loci : list[str], optional
        List of HLA loci to consider (default: ['A', 'B', 'C', 'DRB1', 'DQB1']).
    haplotype_frequency_column : dict[str, str], optional
        Column mapping for haplotype frequency (default: {"frequency": "freq"}).

    Returns
    -------
    float
        CPRA percentage (0-100).

    Example
    -------
    >>> uas = ['A*02:01', 'B*07:02']
    >>> dt_haplo = pd.DataFrame({
    ...     'A': ['A*02:01', 'A*01:01'],
    ...     'B': ['B*07:02', 'B*08:01'],
    ...     'C': ['C*07:01', 'C*07:02'],
    ...     'DRB1': ['DRB1*15:01', 'DRB1*03:01'],
    ...     'DQB1': ['DQB1*06:02', 'DQB1*02:01'],
    ...     'freq': [0.05, 0.03]
    ... })
    >>> cpra = _cpra_haplotype_frequency_single(uas, dt_haplo)
    """

    # Count the number of matches for each haplotype
    haplotype_frequency_loci = haplotype_frequency[loci]

    # For each haplotype, count how many unacceptable antigens match
    n_matches = np.zeros(len(haplotype_frequency))
    for i, row in haplotype_frequency_loci.iterrows():
        n_matches[i] = sum(allele in uas for allele in row.values)

    # Filter haplotypes with at least one match
    has_matches = n_matches > 0
    matching_haplotype = haplotype_frequency[has_matches].copy()
    matching_n = n_matches[has_matches]

    if len(matching_haplotype) == 0:
        return 0.0

    # Calculate S_k terms for inclusion-exclusion principle
    S = np.zeros(6)  # S0 to S5, use S1 to S5

    for k in range(1, 6):  # k = 1 to 5
        if np.any(matching_n >= k):
            binomial_coeffs = np.array(
                [comb(int(n), k, exact=True) if n >= k else 0 for n in matching_n]
            )
            S[k] = np.sum(
                binomial_coeffs
                * matching_haplotype[haplotype_frequency_column["frequency"]]
            )

    # Inclusion-exclusion principle for probability of no match
    prob_no_match = 1 - (S[1] - S[2] + S[3] - S[4] + S[5])

    # Probability of at least one match in a diplotype (two haplotypes)
    cpra = (1 - prob_no_match**2) * 100

    return cpra


def cpra_haplotype_frequency(
    unacceptable_antigen: pd.DataFrame,
    haplotype_frequency: pd.DataFrame,
    loci: list[str] = ["A", "B", "C", "DRB1", "DQB1"],
    unacceptable_antigen_columns: dict[str, str] = {
        "recipient_id": "id",
        "unacceptable_antigen": "uas",
    },
    haplotype_frequency_column: dict[str, str] = {"frequency": "freq"},
) -> pd.DataFrame:
    """
    Calculate CPRA (Calculated Panel Reactive Antibody) for each recipient using
    the haplotype frequency method.

    This function calculates CPRA for each recipient by estimating the probability
    that a random donor has at least one unacceptable antigen, using population
    haplotype frequencies and the inclusion-exclusion principle.

    Parameters
    ----------
    unacceptable_antigen : pd.DataFrame
        Recipient unacceptable antigen data, including at least two columns:
        - recipient ID
        - unacceptable antigen (HLA allele)
    haplotype_frequency : pd.DataFrame
        Haplotype frequency data, including columns for loci and haplotype frequency.
    loci : list[str], optional
        List of HLA loci to consider (default: ['A', 'B', 'C', 'DRB1', 'DQB1']).
    unacceptable_antigen_columns : dict[str, str], optional
        Column mapping for recipient unacceptable antigen data.
    haplotype_frequency_column : dict[str, str], optional
        Column mapping for haplotype frequency (default: {"frequency": "freq"}).

    Returns
    -------
    pd.DataFrame
        DataFrame containing recipient ID and corresponding CPRA percentage.

    Example
    -------
    >>> unacceptable_antigen = pd.DataFrame({
    ...     'id': [1, 1, 2, 2],
    ...     'uas': ['A*02:01', 'B*07:02', 'DRB1*04:01', 'DQB1*03:01']
    ... })
    >>> dt_haplo_freq = pd.DataFrame({
    ...     'A': ['A*02:01', 'A*01:01', 'A*03:01'],
    ...     'B': ['B*07:02', 'B*08:01', 'B*35:01'],
    ...     'C': ['C*07:01', 'C*07:02', 'C*04:01'],
    ...     'DRB1': ['DRB1*15:01', 'DRB1*03:01', 'DRB1*04:01'],
    ...     'DQB1': ['DQB1*06:02', 'DQB1*02:01', 'DQB1*03:01'],
    ...     'freq': [0.4, 0.3, 0.3]
    ... })
    >>> result = cpra_haplotype_frequency(unacceptable_antigen, dt_haplo_freq)
    """

    # Standardize recipient column names
    unacceptable_antigen = unacceptable_antigen.rename(
        columns={
            unacceptable_antigen_columns["recipient_id"]: "id_r",
            unacceptable_antigen_columns["unacceptable_antigen"]: "uas",
        }
    )[["id_r", "uas"]]

    # Group unacceptable antigens by recipient
    grouped_uas = unacceptable_antigen.groupby("id_r")["uas"].apply(list).reset_index()

    # Calculate CPRA for each recipient
    cpra = []
    for _, row in tqdm(
        grouped_uas.iterrows(),
        total=len(grouped_uas),
        desc="Calculating CPRA",
        bar_format=TQDM_FORMAT,
    ):
        id_r = row["id_r"]
        uas_list = row["uas"]
        cpra_value = _cpra_haplotype_frequency_single(
            uas=uas_list,
            haplotype_frequency=haplotype_frequency,
            loci=loci,
            haplotype_frequency_column=haplotype_frequency_column,
        )
        cpra.append(
            {unacceptable_antigen_columns["recipient_id"]: id_r, "cpra": cpra_value}
        )

    return pd.DataFrame(cpra)
