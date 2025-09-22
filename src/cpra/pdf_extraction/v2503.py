"""
Module for extracting LSA PDF files (version 2025.03).
"""

from pathlib import Path

import pandas as pd
import pdfplumber


def extract_pdf_tables(pdf_path):
    """
    Extract all tables from PDF file and return as list of DataFrames.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file

    Returns
    -------
    list of pandas.DataFrame
        List of pandas DataFrames extracted from PDF tables

    Raises
    ------
    FileNotFoundError
        If PDF file doesn't exist
    """
    # Check if file exists
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Error: File '{str(pdf_path)}' does not exist. Please check the filename and path."
        )

    df_list = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Successfully opened PDF file: {pdf_path}")
        print(f"Total pages: {len(pdf.pages)}")

        # Iterate through each page of the PDF
        for i, page in enumerate(pdf.pages):
            print(f"\n--- Processing page {i + 1} ---")

            tables = page.extract_tables()

            if tables:
                print(f"Found {len(tables)} table(s) on page {i + 1}.")
                # Iterate through each table found on current page
                for j, table_data in enumerate(tables):
                    if table_data:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        df_list.append(df)
                    else:
                        print(f"Table {j + 1} found on page {i + 1} is empty.")

            else:
                print(f"No tables found on page {i + 1}.")

    return df_list


def extract_patient_info(df_list, header_keywords=None):
    """
    Extract patient information from DataFrame list and normalize it.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of pandas DataFrames extracted from PDF
    header_keywords : list of str, optional
        Keywords to identify patient info table.
        Default is ["患者姓名", "患者性别", "出生日期"]

    Returns
    -------
    pandas.DataFrame
        Normalized single-row DataFrame with patient information

    Raises
    ------
    ValueError
        If patient information table is not found or format changed
    """
    if header_keywords is None:
        header_keywords = ["患者姓名", "患者性别", "出生日期"]

    columns_list = [df.columns for df in df_list]

    # Find DataFrame indices that contain all keywords simultaneously
    matching_indices = []
    for i, cols in enumerate(columns_list):
        # Check if all keywords appear in column names
        if all(any(kw in str(col) for col in cols) for kw in header_keywords):
            matching_indices.append(i)

    if len(matching_indices) != 1:
        raise ValueError("PDF format changed, please check the extraction logic.")

    # Extract patient information
    info_df = df_list[matching_indices[0]]
    info_list = [c for c in info_df.columns]
    info_list += info_df.values.flatten().tolist()
    info_list = [str(c).strip().replace("\n", "_") for c in info_list if not pd.isna(c)]

    # Process info_list in pairs: first element as key, second as value
    info_dict = {}
    for i in range(0, len(info_list), 2):
        if i + 1 < len(info_list):
            key = info_list[i]
            value = info_list[i + 1]
            info_dict[key] = value

    return pd.DataFrame([info_dict])


def extract_antibody_results(df_list, header_keywords=None):
    """
    Process and combine tables based on keyword matching to extract antibody results.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of pandas DataFrames extracted from PDF
    header_keywords : list of str, optional
        Keywords to identify relevant tables.
        Default is ["特异性", "结果判读"]

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with HLA_I and HLA_II tags

    Raises
    ------
    ValueError
        If expected table structure is not found
    """
    if header_keywords is None:
        header_keywords = ["特异性", "结果判读"]

    columns_list = [df.columns for df in df_list]

    # Find DataFrame indices that contain all keywords simultaneously
    matching_indices = []
    for i, cols in enumerate(columns_list):
        # Check if all keywords appear in column names
        if all(any(kw in str(col) for col in cols) for kw in header_keywords):
            matching_indices.append(i)

    if len(matching_indices) != 2:
        raise ValueError("PDF format changed, please check the extraction logic.")

    # Define index ranges for HLA class I and II sections
    index_list = [
        (matching_indices[0], matching_indices[1] - 1),  # HLA class I section
        (matching_indices[1], len(df_list)),  # HLA class II section
    ]

    df_results = []
    for i_beg, i_end in index_list:
        res = []
        for i, df in enumerate(df_list[i_beg : i_end + 1]):
            if i == 0:
                # First table: use its columns as reference
                columns = [str(c).replace("\n", "_") for c in df.columns]
                new_df = df.copy()
                new_df.columns = columns
                res.append(new_df)
            else:
                # Subsequent tables: treat column headers as data rows
                column_data = [c for c in df.columns]
                all_data = [column_data] + df.values.tolist()
                new_df = pd.DataFrame(all_data, columns=columns)
                res.append(new_df)
        res = pd.concat(res, ignore_index=True).reset_index(drop=True)
        df_results.append(res)

    # Combine HLA class I and II results with tags
    df_results = pd.concat(
        [df_results[0].assign(tag="HLA_I"), df_results[1].assign(tag="HLA_II")],
        ignore_index=True,
    ).reset_index(drop=True)

    return df_results


def extract_unacceptable_antigens(
    df_results, specificity_keywords=None, mfi_keywords=None, mfi_cutoff=750
):
    """
    Extract unacceptable antigens list from processed DataFrame based on MFI cutoff.

    Parameters
    ----------
    df_results : pandas.DataFrame
        Processed DataFrame with HLA antibody data
    specificity_keywords : list of str, optional
        Keywords to identify specificity columns.
        Default is ["Specificity", "特异性"]
    mfi_keywords : list of str, optional
        Keywords to identify MFI columns.
        Default is ["MFI", "荧光中位值"]
    mfi_cutoff : float, optional
        MFI threshold for filtering. Default is 750

    Returns
    -------
    list of str
        List of unacceptable antigen values split by spaces

    Raises
    ------
    ValueError
        If required columns are not found
    """
    if specificity_keywords is None:
        specificity_keywords = ["Specificity", "特异性"]
    if mfi_keywords is None:
        mfi_keywords = ["MFI", "荧光中位值"]

    # Find MFI column index
    mfi_col_indices = [
        i
        for i, col in enumerate(df_results.columns)
        if any(kw in str(col) for kw in mfi_keywords)
    ]
    if len(mfi_col_indices) != 1:
        raise ValueError("MFI column not found, please check the extraction logic.")
    mfi_col_index = mfi_col_indices[0]

    # Find specificity column index
    specificity_col_indices = [
        i
        for i, col in enumerate(df_results.columns)
        if any(kw in str(col) for kw in specificity_keywords)
    ]
    if len(specificity_col_indices) != 1:
        raise ValueError(
            "Specificity column not found, please check the extraction logic."
        )
    specificity_col_index = specificity_col_indices[0]

    # Filter data based on MFI cutoff
    filtered_specificities = (
        df_results[df_results.iloc[:, mfi_col_index].astype(float) > mfi_cutoff]
        .iloc[:, specificity_col_index]
        .values
    )

    # Extract and split unacceptable antigen values
    unacceptable_antigens = []
    for specificity in filtered_specificities:
        unacceptable_antigens.extend(specificity.split(" "))

    return unacceptable_antigens
