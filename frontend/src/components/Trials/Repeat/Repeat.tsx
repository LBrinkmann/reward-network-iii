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
                Repeat the solution of the previous player
            </Typography>
            <NetworkTrial showComment={true} teacherId={teacherId} playerTotalPoints={playerTotalPoints}
            showTotalPoints={false}/>
        </>
    );

}


export default Repeat;