# ISIC 2019
The dataset used in this repo comes from the [ISIC2019 challenge](https://challenge.isic-archive.com/landing/2019/) and the [HAM1000 database](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
We do not own the copyright of the data, everyone using those datasets should abide by their licences (see below) and give proper attribution to the original authors.

## Dataset Fetch
The data is downloaded with using FLamby package. repo(https://github.com/owkin/FLamby/tree/main)

## Dataset description
The following table provides a data sheet:

|                   | Dataset description
| ----------------- | -----------------------------------------------------------------------------------------------
| Description       | Dataset from the ISIC 2019 challenge, we keep images for which the datacenter can be extracted.
| Dataset           | 23,247 images of skin lesions ((9930/2483), (3163/791), (2691/672), (1807/452), (655/164), (351/88))
| Centers           | 6 centers (BCN, HAM_vidir_molemax, HAM_vidir_modern, HAM_rosendahl, MSK, HAM_vienna_dias)
| Task              | Multiclass image classification

### License
The [full licence](https://challenge.isic-archive.com/data/#2019) for ISIC2019 is CC-BY-NC 4.0.

In order to extract the origins of the images in the HAM10000 Dataset (cited above), we store in this repository a copy of [the original HAM10000 metadata file](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
Please find attached the link to the [full licence and dataset terms](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T&version=3.0&selectTab=termsTab) for the HAM10000 Dataset.

Please first accept the licences on the HAM10000 and ISIC2019 dataset pages before going
through the following steps.

### Ethics
As per the [Terms of Use](https://challenge.isic-archive.com/terms-of-use/) of the
[website](https://challenge.isic-archive.com/) hosting the dataset,
one of the requirements for this datasets to have been hosted is that it is
properly de-identified in accordance with the
applicable requirements and legislations.