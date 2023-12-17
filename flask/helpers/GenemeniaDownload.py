import requests
from bs4 import BeautifulSoup
import os

# URL of the directory containing files to download
url = 'https://genemania.org/data/current/Homo_sapiens/'

def download_files(base_url):
    try:
        # Fetch the content from the URL
        response = requests.get(base_url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all the <a> tags to identify downloadable files
        links = soup.find_all('a')

        # Directory to save the files
        download_dir = 'C:/Users/kuday/Desktop/CS 319/cs491-2-web/flask/data'
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download each file
        for link in links:
            file_url = link.get('href')
            if file_url and file_url.endswith('.txt'):  # Check if it's a downloadable .txt file
                full_url = base_url + file_url
                file_response = requests.get(full_url)
                file_response.raise_for_status()

                # Save the file
                with open(os.path.join(download_dir, file_url), 'wb') as file:
                    file.write(file_response.content)
                print(f"Downloaded {file_url}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to start downloading files
download_files(url)
