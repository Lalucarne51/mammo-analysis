from tcia_utils import nbia
import dicom2jpg


def download_and_preprocess():

    data = "../data/data_raw/"
    dir_data_processed = "../data/data_processed/"
    manifest = "../manifest/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia"

    # converts manifest to list of UIDs
    uids = nbia.manifestToList(manifest)

    buckt_img = 0
    for pict_dcm in uids:
        for i in range(buckt_img, buckt_img+10):
            nbia.downloadSeries('../data/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia', input_type = "manifest", number=buckt_img, format = "df",path=data)
            dicom2jpg.dicom2jpg(data, target_root=dir_data_processed)
            buckt_img = buckt_img+10

download_and_preprocess()
