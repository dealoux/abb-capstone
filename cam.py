from pypylon import pylon
import cv2

tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
for device in devices:
    print(device.GetFriendlyName())

camera = pylon.InstantCamera()
camera.Attach(tl_factory.CreateFirstDevice())
camera.Open()
camera.StartGrabbing(1)

while True:
    grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
    if grab.GrabSucceeded():
        img = grab.GetArray()
        print(f'Size of image: {img.shape}')
        # img_bgr = cv2.cvtColor(img,cv2.COLOR_BAYER_GB2BGR)
        # img_bgr = cv2.cvtColor(img,cv2.COLOR_YUV2BGR_Y422)
        converter = pylon.ImageFormatConverter()
        # dinh dang dau ra cho hinh anh la BGR - voi 8 bit cho moi kenh
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        img_bgr = converter.Convert(grab).GetArray()
        cv2.putText(img_bgr,"LONG DUT",(100,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        img_resize = cv2.resize(img_bgr,None,fx=0.5,fy=0.5)



        cv2.imshow("Anh goc tư camera",img_resize)


        cv2.waitKey(1)

camera.Close()