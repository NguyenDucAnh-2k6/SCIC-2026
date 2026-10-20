import urllib.request
import re

urls = [
    'https://www.sociopatterns.org/datasets/primary-school-temporal-network-data/',
    'https://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/',
    'https://www.sociopatterns.org/datasets/high-school-dynamic-contact-networks/'
]

for url in urls:
    print("Checking:", url)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req).read().decode('utf-8')
        for link in re.findall(r'href=["\']([^"\']+)["\']', html):
            if any(ext in link for ext in ['.csv', '.tsv', '.gz', '.zip', '.tar']):
                print("  Found:", link)
    except Exception as e:
        print("  Error:", e)
