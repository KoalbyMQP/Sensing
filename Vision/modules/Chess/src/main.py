
import cv_model
import localization
import depthai as dai


def main():

    MODEL_PATH = (
    "/Users/azieldawit/Desktop/School/WPI/MQP/Sensing/Vision/modules/Chess/models/version2_compressed/version2_compressed.rvc2.tar.xz"
    )   
    device = dai.Device(dai.DeviceInfo(None)) if None else dai.Device()
    
    
    model = cv_model.Model(MODEL_PATH, device)
    model.liveInference()

    return



if __name__ == "__main__":
    main()