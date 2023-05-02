from benchmark_viper import VIPER

def main():
    viper = VIPER()
    # viper.DownloadAndUnpack("/srv/share4/datasets/VIPER_Flowv2/archive", "/srv/share4/datasets/VIPER_Flowv2/unpacked", None)
    # viper.ConvertToKittiFormat("/srv/share4/datasets/VIPER_Flowv2/archive", None, "/srv/share4/datasets/VIPER_Flowv2/train", "/srv/share4/datasets/VIPER_Flowv2/test")
    viper.ConvertToKittiFormat2("/srv/share4/datasets/VIPER_Flowv3/val", "/srv/share4/datasets/VIPER_Flowv3/val")
    # print("starting")
    # viper.ConvertToKittiFormat2("/srv/share4/datasets/VIPER_Flowv3/train", "/srv/share4/datasets/VIPER_Flowv3/train")
    


main()