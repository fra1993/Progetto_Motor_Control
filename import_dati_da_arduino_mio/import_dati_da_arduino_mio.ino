volatile unsigned char TXBuf[1];
char header='_';
int ADC_Value;
void setup() {
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:
  for(int CurrentCh=0;CurrentCh<6;CurrentCh++){

    ADC_Value = analogRead(CurrentCh);
    Serial.print(header);
    Serial.print(ADC_Value);
    }
}
