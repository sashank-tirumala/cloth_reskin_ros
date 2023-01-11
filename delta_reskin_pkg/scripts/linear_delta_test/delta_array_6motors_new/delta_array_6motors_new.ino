#include "linear_actuator.pb.h"
#include "pb_common.h"
#include "pb.h"
#include "pb_encode.h"
#include "pb_decode.h"
#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include <Adafruit_ADS1X15.h>
#include<math.h>

#define NUM_MOTORS 6
//################################## Feather MC and ADC Libraries INIT #####################3
// $$$$$$$$$$$ FOR RIGHT SETUP $$$$$$$$$$$$$$$$$
Adafruit_MotorShield MC0 = Adafruit_MotorShield(0x60);
Adafruit_MotorShield MC1 = Adafruit_MotorShield(0x61);
// $$$$$$$$$$$ FOR LEFT SETUP $$$$$$$$$$$$$$$$$$
//Adafruit_MotorShield MC0 = Adafruit_MotorShield(0x61);
//Adafruit_MotorShield MC1 = Adafruit_MotorShield(0x60);

Adafruit_ADS1015 ADC1;
Adafruit_ADS1015 ADC0;

Adafruit_DCMotor *MC0_M1 = MC0.getMotor(1);
Adafruit_DCMotor *MC0_M2 = MC0.getMotor(2);
Adafruit_DCMotor *MC0_M3 = MC0.getMotor(3);
Adafruit_DCMotor *MC1_M1 = MC1.getMotor(1);
Adafruit_DCMotor *MC1_M2 = MC1.getMotor(2);
Adafruit_DCMotor *MC1_M3 = MC1.getMotor(3);

Adafruit_DCMotor* motors[NUM_MOTORS] = {
                                        MC0_M1,MC0_M2,MC0_M3,// 1st robot
                                        MC1_M1,MC1_M2,MC1_M3,// 2nd robot
//                                        MC2_M3,MC2_M4,MC1_M3,// 3rd robot
//                                        MC1_M4,MC0_M3,MC0_M4,// 4th robot
                                        };


Adafruit_ADS1015* adcs[NUM_MOTORS] = {
                                      &ADC0, &ADC0, &ADC0,//1st robot
                                      &ADC1, &ADC1, &ADC1,//2nd robot
//                                      &ADC0, &ADC0, &ADC1,//3rd robot
//                                      &ADC1, &ADC2, &ADC2,//4th robot
                                      };

int channels[NUM_MOTORS] = {
                            0,1,2,//1st robot
                            0,1,2,//2nd robot
//                            2,3,2,//3rd robot
//                            3,2,3,//4th robot
                            };

                            
//Adafruit_ADS1015* adcs[NUM_MOTORS] = {&ADC1};
//Adafruit_DCMotor* motors[NUM_MOTORS] = {MC1_M1};
//int channels[NUM_MOTORS] = {0};


//##################################### GLOBAL VARIABLES ###################################3
const int numChars = 128;
float pi = 3.1415926535;
float p = 300.0;
float i_pid = 0.1;
float d = 3.75;

uint8_t input_cmd[numChars];
bool newData = false;

lin_actuator command = lin_actuator_init_zero;
static boolean recvInProgress = false;
static byte ndx = 0;
char startMarker = 0xA6;
char endMarker = 0xA7;
char confMarker = '~';
unsigned long current_arduino_time;
unsigned long last_arduino_time;
float time_elapsed;

float joint_positions[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
float new_joint_positions[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

float position_threshold = 0.0008;
float joint_errors[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

// We need last_joint_errors to compute PID control value for d. 
float last_joint_errors[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
// We need total_joint_errors to compute PID control value for i. 
float total_joint_errors[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

int motor_val[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
bool is_movement_done = false;

//################################# SETUP AND LOOP FUNCTIONS ################################3
void setup() {
  // put your setup code here, to run once:
  Serial1.begin(57600);
  while (!Serial1)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens
  
  // set all the base dc motor control pins to outputs
  MC0.begin();
  MC1.begin();
///  // start all the ADCs
  ADC1.begin(0x49);
  ADC0.begin(0x48);

//  ADC2.setGain(GAIN_ONE);
  ADC1.setGain(GAIN_ONE);
  ADC0.setGain(GAIN_ONE);

  // disable dc motors by setting their enable lines to low
  for(int i=0; i<NUM_MOTORS; i++){
    motors[i]->setSpeed(150);
    motors[i]->run(RELEASE);
    delay(10);
  }
  
  readJointPositions();
//  resetJoints();
//  stop();
}

void loop() {
  // put your main code here, to run repeatedly:
  recvWithStartEndMarkers();
  if (newData == true) {
    if (decodeNanopbData()){
      writeJointPositions();
    }
    newData = false;
    ndx = 0;
  }
//  writeJointPositions();
}

//############################# READ / WRITE JOINT POSITIONS #######################################3
void readJointPositions(){
  for(int i = 0; i < NUM_MOTORS; i++){
    motor_val[i] = adcs[i]->readADC_SingleEnded(channels[i]); 
    joint_positions[i] = motor_val[i] * 0.00006; // 100mm / 1650
  }
}

void writeJointPositions(){
  bool reached_point = false;
  last_arduino_time = millis();
  is_movement_done = false;
  while (!reached_point){
    current_arduino_time = millis();
    time_elapsed = float(current_arduino_time - last_arduino_time) / 1000.0;
    readJointPositions();
    reached_point = true;
    for(int i = 0; i < NUM_MOTORS; i++){
      joint_errors[i] = joint_positions[i] - new_joint_positions[i];
      
      float pid = p * joint_errors[i] + i_pid * total_joint_errors[i] + d * (joint_errors[i] - last_joint_errors[i]) / time_elapsed;
      
      if(joint_errors[i] > position_threshold){
        int motor_speed = (int)(min(max(0.0, pid), 1.0) * 255.0);
        reached_point = reached_point && false;
        motors[i]->setSpeed(motor_speed);
        motors[i]->run(BACKWARD);
        total_joint_errors[i] += joint_errors[i];
      }
      else if(joint_errors[i] < -position_threshold){
        int motor_speed = (int)(min(max(-1.0, pid), 0.0) * -255.0);
        reached_point = reached_point && false;
        motors[i]->setSpeed(motor_speed);
        motors[i]->run(FORWARD);
        total_joint_errors[i] += joint_errors[i];
      }
      else{
        reached_point = reached_point && true;;
        motors[i]->setSpeed(0);
        motors[i]->run(RELEASE);
        total_joint_errors[i] = 0.0;
      }
    }
    last_arduino_time = current_arduino_time;
  }
  for(int i = 0; i < 12; i++){
      motors[i]->setSpeed(0);
      motors[i]->run(RELEASE);
      total_joint_errors[i] = 0.0;
  }
  is_movement_done = true;
   
//  Serial.println("~ Moved to New Position");
}

//########################### STOP OR RESET FUNCTIONS ######################################3
void resetJoints(){
  for(int i = 0; i < NUM_MOTORS; i++)
  {
    new_joint_positions[i] = 0.0;
  }
  writeJointPositions();
}

void stop(){
  // Turn off all motors
  for(int i = 0; i < NUM_MOTORS; i++)
  {
    motors[i]->run(RELEASE);
  }
}

//############################ SERIAL COMM FUNCTIONS #######################################3
void recvWithStartEndMarkers() {
  byte rc;
  while (Serial1.available() > 0 && newData == false) {
    rc = Serial1.read();

    if (recvInProgress == true) {
      if (rc == endMarker) {
        Serial.println(rc);
        delay(10);
        rc = Serial1.read();
        if (rc == confMarker) {
          Serial.println(rc);
          delay(10);
          rc = Serial1.read();
          Serial.println(rc);
          if (rc == confMarker) {
            input_cmd[ndx] = '\0'; // terminate the string
            recvInProgress = false;
            newData = true;
            for (int i=0; i<ndx; i++){
              Serial.print(input_cmd[i]);Serial.print(" ");
            }
            Serial.println();
            
          }else{
            input_cmd[ndx] = rc;
            ndx++;
            if (ndx >= numChars) {
              ndx = numChars - 1;
            }
          }
        } else{
          input_cmd[ndx] = rc;
          ndx++;
          if (ndx >= numChars) {
            ndx = numChars - 1;
          }
        }
      }
      else {
        input_cmd[ndx] = rc;
        ndx++;
        if (ndx >= numChars) {
          ndx = numChars - 1;
        }
      }
    }

    else if (rc == startMarker) {
      delay(10);
      rc = Serial1.read();
      if (rc == confMarker) {
        delay(10);
        rc = Serial1.read();
        if (rc == confMarker) {
          recvInProgress = true;
        }
      }
    }
  }
}
///

bool decodeNanopbData(){
  lin_actuator message = lin_actuator_init_zero;
  pb_istream_t istream = pb_istream_from_buffer(input_cmd, ndx);
  bool ret = pb_decode(&istream, lin_actuator_fields, &message);
  if (ret){
    for (int i=0; i<NUM_MOTORS; i++){
      new_joint_positions[i] = message.joint_pos[i];
    }
  }
  else{
    ret = false;
  }
  return ret;
}
