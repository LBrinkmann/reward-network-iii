import React, { FC } from "react";

import { Box, Button, CardMedia, Grid, Typography } from "@mui/material";
import CheckIcon from "@mui/icons-material/Check";

import instructions from "./InstructionContent";
import imgNetExample from "../../../images/net_practice_example.png";

interface InstructionInterface {
  endTrial: (data: any) => void;
  instructionType: keyof typeof instructions;
}

export const Instruction: FC<InstructionInterface> = ({
  endTrial,
  instructionType,
}) => {
  const onClickHandler = () => endTrial({ moves: [] });

  return (
    <Grid container spacing={4}>
      <Grid item xs={12}>
        <Box
          sx={{ width: "90%", maxWidth: "600px" }}
          m="auto" // box margin auto to make box in the center
          style={{ maxHeight: "80vh", overflow: "auto" }} //maxHeight: 300,
          p={3} // box padding
        >
          {instructionType === "welcome" && <Welcome />}
          {instructionType !== "welcome" &&
            instructions[instructionType].map((paragraph, index) => (
              <Typography key={index} variant="body1" align="justify" paragraph>
                {paragraph}
              </Typography>
            ))}

          <Grid item xs={12} textAlign={"center"} p={2}>
            <Button
              variant="contained"
              color="success"
              onClick={onClickHandler}
              startIcon={<CheckIcon />}
            >
              ️ Continue
            </Button>
          </Grid>
        </Box>
      </Grid>
    </Grid>
  );
};

export default Instruction;

const Welcome: FC = () => {
  return (
    <>
      <Typography gutterBottom variant="h5" align="center">
        Welcome to the experiment!
      </Typography>
      <Grid container direction="column">
        <Grid container direction="row">
          <Grid item xs={6}>
            <Typography variant="body1" align="justify" paragraph>
              You will be working on a task where your goal is to navigate through
              a series of networks like the one depicted on the right. Depending
              on the moves you choose, you can earn more or less points. These
              points will be converted into bonus payments, so your decisions in
              the main task will have real financial consequences for you.
            </Typography>
            <Typography variant="body1" align="justify" paragraph>
              To prepare you for the main task, we will now explain the network
              in more detail.
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <CardMedia
              component="img"
              image={imgNetExample}
              style={{ maxWidth: "400px" }}
              alt="Example network"
            />
          </Grid>
        </Grid>
      </Grid>
    </>
  );
};
