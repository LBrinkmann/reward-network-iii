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
              Our experiment involves networks like the one depicted on the right. Depending
              on the moves you choose to navigate through the networks, you can earn more or less points. These
              points will be converted into bonus payments -- so your decisions in
              this experiment will have real financial consequences for you.
            </Typography>
            <Typography variant="body1" align="justify" paragraph>
              We will now explain the network task in more detail. 
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
