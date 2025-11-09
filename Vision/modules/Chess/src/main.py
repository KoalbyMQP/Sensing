
import cv_model
import localization
import depthai as dai


def main():

    MODEL_PATH = (
    "/Users/azieldawit/Desktop/School/WPI/MQP/Sensing/Vision/modules/Chess/models/version2_compressed/version2_compressed.rvc2.tar.xz"
    )   
    device = dai.Device(dai.DeviceInfo(None)) if None else dai.Device()
    
    
    model = cv_model.Model(MODEL_PATH, device)
    # predict creates its own pipeline, no need to call requestImage first
    model.predict()

    return



if __name__ == "__main__":
    main()