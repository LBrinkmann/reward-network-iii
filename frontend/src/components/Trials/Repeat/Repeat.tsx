import NetworkTrial from "../NetworkTrial";
import React, {FC} from "react";
import {Typography} from "@mui/material";


export interface IRepeat {
    solution: number[];
    teacherId: number;
    playerTotalPoints: number;
}

const Repeat: FC<IRepeat> = ({solution, teacherId, playerTotalPoints}) => {
    return (
        <>
            <Typography variant="h3" align='center'>
                Collect points by correctly repeating the path of player {teacherId}
            </Typography>
            <NetworkTrial showComment={true} teacherId={teacherId} playerTotalPoints={playerTotalPoints}
            showTotalPoints={false}/>
        </>
    );

}


export default Repeat;