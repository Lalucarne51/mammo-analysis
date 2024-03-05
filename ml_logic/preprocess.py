from tcia_utils import nbia
import dicom2jpg

def download_and_preprocess():

    data = "../data/"
    dir_data_processed = "../data_processed/"
    # Data download:
    download_serie = nbia.getSeries(collection = "CBIS-DDSM", format = "df", path= data)

    buckt_img = 0

    for pict_dcm in download_serie:
        pictures = nbia.downloadSeries(download_serie, number = buckt_img, path=data)
        dicom2jpg.dicom2jpg(pictures, target_root=dir_data_processed)
        buckt_img = buckt_img+1
