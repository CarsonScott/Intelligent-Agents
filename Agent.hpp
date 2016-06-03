#ifndef AGENT_HPP_INCLUDED
#define AGENT_HPP_INCLUDED

class Agent: public sf::CircleShape{
    float speed;
    float rotate_speed;
    float distance;
    PVector velocity;
    std::vector<Sensor> sensors;
    std::vector<PVector> offsets;
    Array distances;
    Array states;
    Network network;
    sf::FloatRect old_rect;
    sf::FloatRect new_rect;
public:

    Agent(int sensor_count){
        setRadius(20);
        setOrigin(getRadius(), getRadius());

        distance = random(150, 10);

        for(int i = 0; i < sensor_count; i++){
            float angle = atan2(i, 1)-45;
            PVector offset;
            offset.x = i*random(100)*cos(angle);
            offset.y = i*random(100)*sin(angle);
            offsets.push_back(offset);

            distances.push_back(random(distance));
            addSensor(offset);
        }

        network.create(sensor_count+4, sensor_count*2, 6);
        speed = 50*sensor_count;
        rotate_speed = 100;
        velocity = PVector(0, 0);
        old_rect = getGlobalBounds();
    }

    void addSensor(PVector offset){
        Sensor sensor;
        sensor.setPosition(getPosition() + offset);
        sensor.setFillColor(sf::Color(255, 255, 255, 150));

        sensor.setRadius(5);
        sensor.setOrigin(sensor.getRadius(), sensor.getRadius());

        sensors.push_back(sensor);
        states.push_back(float());
    }

    void detect(PVector target){
        for(int i = 0; i < sensors.size(); i++){
            states[i] = 100*1/getDistance(sensors[i].getPosition(), target);
        }
    }

    void update(int reward, float dt){
        old_rect = getGlobalBounds();

        Array input = states;
        input.push_back(reward);
        input.push_back(getRotation());
        input.push_back(velocity.x);
        input.push_back(velocity.y);

        network.update(input, reward);
        Array output = network.getOutputs();

        float angle = getRotation();
        angle += -output[0]*rotate_speed;
        angle += output[1]*rotate_speed;
        setRotation(angle);

        for(int i = 0; i < sensors.size(); i++){
            float dist = distances[i];

            PVector offset = getOffset(getPosition(), sensors[i].getPosition());
            float r = atan2(offsets[i].y, offsets[i].x)*180/PI;

            PVector pos = getPosition();
            pos.x += dist*cos(angle+r);
            pos.y += dist*sin(angle+r);

            sensors[i].setPosition(pos);
        }

        velocity.x += output[2]*speed*cos(angle)*dt;
        velocity.y += output[3]*speed*sin(angle)*dt;
        move(mult(velocity, dt));

        if(abs(velocity.x) > 0){
            velocity.x += -dt*speed*output[4]*.5*(velocity.x/abs(velocity.x));
        }
        if(abs(velocity.y) > 0){
            velocity.y += -dt*speed*output[5]*.5*(velocity.y/abs(velocity.y));
        }
        new_rect = getGlobalBounds();
    }

    void draw(sf::RenderWindow& window){
        window.draw(*this);
        for(int i = 0; i < sensors.size(); i++){
            window.draw(sensors[i]);
        }
    }

    void train(){
        network.learn();
    }

    void reset(){
        setRotation(random(360)-180);
        velocity = PVector(0, 0);
    }
};

#endif // AGENT_HPP_INCLUDED
