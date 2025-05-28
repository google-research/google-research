# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matlabplotlib.pyplot as plt

def parse_proceedings(year, year_url, year_url_get, style_format=None):
    """
    Parses the System Dynamics Proceedings website for a given year and extracts paper information.

    Args:
      year (str): The year of the proceedings.
      year_url (str): The URL of the proceedings for the given year.
      year_url_get (str): The base URL for retrieving individual papers and abstracts.
      style_format (str, optional): The CSS style format used for identifying paper entries.
                                     Defaults to None.

    Returns:
      pd.DataFrame: A DataFrame containing the extracted paper information.
    """

    sd_papers_df = pd.DataFrame(columns=['year', 'authors', 'title', 'abstract', 'paper_path', 'supporting_path'])
    year_response = requests.get(year_url)

    if year_response.status_code == 200:
        year_soup = BeautifulSoup(year_response.content, 'html.parser')

        # Determine how to find paper entries based on style_format
        if style_format:
            p_tag = year_soup.find_all('p', style=style_format)
        else:
            p_tag = year_soup.find_all('p', {'class':'Papers'})  # For Format 2

        for p in p_tag:
            try:
                name = p.contents[0]
                title = p.contents[1].get_text() if len(p.contents) > 1 else ''
                abstract_text = ''
                paper_filepath = ''
                supporting_filepath = ''

                for i_ in p.find_all('a'):
                    if i_.get_text() == 'Abstract':
                        abstract_text = extract_abstract(year_url_get, i_.get('href'))
                    elif i_.get_text() == 'Paper':
                        paper_filepath = download_pdf(year, year_url_get, i_.get('href'))
                    elif i_.get_text() == 'Supporting':
                        supporting_filepath = download_pdf(year, year_url_get, i_.get('href'))

                sd_papers_df = pd.concat([sd_papers_df,
                                          pd.DataFrame([
                                              {'year': year,
                                               'authors': name,
                                               'title': title,
                                               'abstract': abstract_text,
                                               'paper_path': paper_filepath,
                                               'supporting_path': supporting_filepath
                                              }]
                                              )], ignore_index=True)
            except Exception as e:
                print(f"Error parsing: {e}")
                print(year_url_get)
                print(p)
                # raise

        return sd_papers_df
    else:
        print('Error parsing URL')
        return sd_papers_df

def extract_abstract(year_url_get, abstract_link):
    """
    Extracts the abstract text from the given URL.

    Args:
      year_url_get (str): The base URL for retrieving the abstract.
      abstract_link (str): The relative link to the abstract page.

    Returns:
      str: The extracted abstract text.
    """
    abstract_resp = requests.get(year_url_get + abstract_link)
    if abstract_resp.status_code == 200:
        abstract_soup = BeautifulSoup(abstract_resp.content, 'html.parser')
        abstract_p = abstract_soup.find_all('p')

        # Different formats have slightly different abstract structures
        for i, p_ in enumerate(abstract_p):
            if 'Abstract for' in p_.get_text():
                return abstract_p[i+1].get_text()  # Format 1, 2, 4

        return abstract_p[1].get_text()  # Format 3 (fallback)
    else:
        return 'Error - Abstract link: ' + year_url_get + abstract_link

def download_pdf(year, year_url_get, pdf_link):
    """
    Downloads a PDF file from the given URL and saves it to a directory.

    Args:
      year (str): The year of the proceedings.
      year_url_get (str): The base URL for retrieving the PDF.
      pdf_link (str): The relative link to the PDF file.

    Returns:
      str: The file path of the downloaded PDF.
    """
    header = {'User-Agent': 'scrapping_script/1.0'}
    paper_resp = requests.get(year_url_get + pdf_link, headers=header)
    if paper_resp.status_code == 200:
        directory = f"/content/sd_papers/{year}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, os.path.basename(pdf_link))
        with open(filepath, "wb") as pdf_file:
            pdf_file.write(paper_resp.content)
        return filepath
    else:
        return 'Error - PDF link: ' + year_url_get + pdf_link

def visualize_data(sd_papers_df):
    """
    Visualizes the distribution of papers and abstracts over the years.

    Args:
      sd_papers_df (pd.DataFrame): DataFrame containing the extracted paper information.
    """
    year_counts = sd_papers_df['year'].value_counts().sort_index()
    abstract_counts = sd_papers_df[sd_papers_df['abstract'] != ''].groupby('year')['abstract'].count()
    paper_counts = sd_papers_df[sd_papers_df['paper_path'] != ''].groupby('year')['paper_path'].count()

    plt.figure(figsize=(12, 6))

    plt.plot(year_counts.index, year_counts.values, label='Total Papers', marker='o')
    plt.plot(abstract_counts.index, abstract_counts.values, label='Abstracts', marker='x')
    plt.plot(paper_counts.index, paper_counts.values, label='Paper PDFs', marker='s')

    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Number of Papers, Abstracts, and Paper PDFs per Year')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    base_url = 'https://proceedings.systemdynamics.org'
    response = requests.get(base_url)
    if response.status_code != 200:
        print('Error fetching the base URL')
        exit()

    soup = BeautifulSoup(response.content, 'html.parser')
    proceedings_links = [a['href'] for a in soup.find_all('a', href=lambda href: href and 'index.html' in href)]

    # Define year lists with more descriptive names
    format_no_class_list = ['2020', '2021', '2022', '2023']
    format_with_class_list = ['2018', '2019']
    format_compact_list = ['2005', '2006', '2007']
    format_standard_list = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

    sd_papers_df = pd.DataFrame(columns=['year', 'authors', 'title', 'abstract', 'paper_path', 'supporting_path'])

    for link in proceedings_links:
        print(link)
        year_url = base_url + '/' + link
        year = link.split('/')[0]

        if year in format_no_class_list:
            year_url_get = base_url + "/" + year + "/"
            df = parse_proceedings(year, year_url, year_url_get, "margin-left: .5in; text-indent: -.5in; margin-bottom: -.12in")
        elif year in format_with_class_list:
            year_url_get = base_url + "/" + year + "/" + "proceed" + "/"
            df = parse_proceedings(year, year_url, year_url_get)  # No style_format needed for Format 2
        elif year in format_compact_list:
            year_url_get = base_url + "/" + year + "/" + "proceed" + "/"
            df = parse_proceedings(year, year_url, year_url_get, "margin-left: 40; text-indent:-40; margin-bottom:-10")
        elif year in format_standard_list:
            year_url_get = base_url + "/" + year + "/" + "proceed" + "/"
            df = parse_proceedings(year, year_url, year_url_get, "margin-left: .5in; text-indent: -.5in; margin-bottom: -.12in")
        else:
            df = None
            print(f"Unknown format for year: {year}")

        if df is not None:
            sd_papers_df = pd.concat([sd_papers_df, df], ignore_index=True)

    # Save the dataframe
    sd_papers_df.to_pickle("sd_papers.pkl")

    # Visualize the papers distribution by year
    visualize_data(sd_papers_df)
