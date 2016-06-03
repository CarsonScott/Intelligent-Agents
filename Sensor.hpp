#ifndef SENSOR_HPP_INCLUDED
#define SENSOR_HPP_INCLUDED

class Sensor: public sf::CircleShape{
    bool state;
public:
    void setState(bool s){
        state = s;
    }

    bool getState(){
        return state;
    }
};

#endif // SENSOR_HPP_INCLUDED
