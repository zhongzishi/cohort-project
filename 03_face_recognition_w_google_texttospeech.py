''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import os
#from playsound import playsound
from google.cloud import texttospeech
import vlc

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/zishi.zhong/Documents/json/tm-datastore-62ed021e8e3b.json"
client = texttospeech.TextToSpeechClient()
if not os.path.exists("mp3"):
    os.mkdir("mp3")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);



font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Zishi', 'JP']
#mp3s = {"None":"sound0.mp3","Zishi":"sound1.mp3","Ali":"sound2.mp3"}
speeches = {"None":"text0","Zishi":"Les quelque 1800 employés de soutien syndiqués de l'Université du Québec à Montréal (UQAM) mèneront mercredi une deuxième journée de grève d’affilée, en cette semaine de rentrée universitaire.","JP":"Un investissement de 107 millions de dollars pour développer le transport de marchandises à l’aéroport de Mirabel a été officiellement annoncé par le ministre fédéral des Transports, Marc Garneau, et par Aéroports de Montréal (ADM), mardi matin."}

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

identification = None
user_id = None
while identification == None:

    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence_display = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence_display = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_display), (x+5,y+h-5), font, 1, (255,255,0), 1)
        if round(100 - confidence) > 50 and confidence < 100:
            identification = True
			#playsound(mp3s.get(user_id))
            synthesis_input = texttospeech.types.SynthesisInput(text=speeches.get(id))
            voice = texttospeech.types.VoiceSelectionParams(
					language_code='fr-CA',
					ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)
            audio_config = texttospeech.types.AudioConfig(
                    audio_encoding=texttospeech.enums.AudioEncoding.MP3)
            response = client.synthesize_speech(synthesis_input, voice, audio_config)
            mp3filepath = "mp3/"+id+".mp3"
            with open(mp3filepath, 'wb') as out:
                out.write(response.audio_content)
            vlc.MediaPlayer(mp3filepath).play()
            break
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
