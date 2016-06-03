#ifndef GRID_HPP_INCLUDED
#define GRID_HPP_INCLUDED

class Grid: public std::vector<std::vector<bool>>{
    PVector block_size;
public:
    Grid(PVector size, PVector block){
        for(int i = 0; i < size.x; i++){
            push_back(std::vector<bool>());
            for(int j = 0; j < size.y; j++){
                back().push_back(0);
            }
        }

        block_size = block;
    }

    PVector getBlockSize(){
        return block_size;
    }

    bool inRange(PVector index){
        if(index.x >= 0 && index.y >= 0){
            if(index.x < size() && index.y < at(0).size()){
                return true;
            }
        }
        return false;
    }

    bool getCellState(PVector index){
        return at(index.x).at(index.y);
    }
};

#endif // GRID_HPP_INCLUDED
