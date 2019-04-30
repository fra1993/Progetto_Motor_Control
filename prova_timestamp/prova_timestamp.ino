#include <Time.h>
#define PACKETLEN 4

unsigned long StartTime;  
unsigned long CurrentTime;
unsigned long ElapsedTime;

 void setup() {
  Serial.begin(57600);
  StartTime = micros();
  
  
}

void loop() {
  CurrentTime = micros();
  ElapsedTime=CurrentTime-StartTime;
  Serial.write(CurrentTime);
  delay(300);
    
  }
