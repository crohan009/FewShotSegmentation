import json

# from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
# from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.ade20k_loader_zhou import ADE20KLoader_Zhou
from ptsemseg.loader.ade20k_few_shot_loader import ADE20KFewShotLoader
# from ptsemseg.loader.mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
# from ptsemseg.loader.cityscapes_loader import cityscapesLoader
# from ptsemseg.loader.nyuv2_loader import NYUv2Loader
# from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
# from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        # "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "ade20k_zhou": ADE20KLoader_Zhou,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "ade20k_few_shot": ADE20KFewShotLoader
        # "cityscapes": cityscapesLoader,
        # "nyuv2": NYUv2Loader,
        # "sunrgbd": SUNRGBDLoader,
        # "vistas": mapillaryVistasLoader,
    }[name]
