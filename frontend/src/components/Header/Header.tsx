import React, {FC} from "react";
import {AppBar, Toolbar, Typography, Box} from "@mui/material";
import useSessionContext from "../../contexts/SessionContext";
import useNetworkContext from "../../contexts/NetworkContext";
import {TRIAL_TYPE} from "../../components/Trials/ExperimentTrial";


interface IHeader {
    showTip?: boolean;
    showTutorial?: boolean;
    title?: string;
    onTutorialClose?: () => void;
}

const Header: FC<IHeader> = (props) => {
    const {showTip= true, showTutorial=false, title='', onTutorialClose} = props;
    const {sessionState, sessionDispatcher} = useSessionContext();
    const {networkState, networkDispatcher} = useNetworkContext();

    const totalPoints = sessionState.totalPoints +
    ([TRIAL_TYPE.INDIVIDUAL, TRIAL_TYPE.REPEAT].includes(sessionState.currentTrialType) && !sessionState.isPractice ? networkState.points : 0);


    return (
        <Box sx={{flexGrow: 1, height: 80}}>
            <AppBar position="static">
                <Toolbar>
                    <Typography variant="h6" sx={{flexGrow: 1}}>
                        {title}
                    </Typography>
                    <Typography variant="h6">
                        Total Points: {totalPoints}
                    </Typography>
                </Toolbar>
            </AppBar>
        </Box>
    );
};

export default Header;
