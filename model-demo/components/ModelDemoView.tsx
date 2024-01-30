import { useEffect, useState } from "react";
import { Suggestion } from "../models/Suggestion";
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  LinearProgress,
  Stack,
  Typography,
} from "@mui/material";
import Reconstruction from "./Reconstruction";
import { Encoding, N_TIME_STEPS, randomEncoding } from "../models/Encoding";
import SegmentEditor from "./SegmentEditor";
import { Equalizer, Refresh } from "@mui/icons-material";

interface ModelDemoViewProps {
  nReconstructions: number;
}

const ModelDemoView: React.FC<ModelDemoViewProps> = ({ nReconstructions }) => {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);

  const [randomPattern, setRandomPattern] = useState<Encoding | undefined>(
    undefined
  );

  const fetchSuggestions = () => {
    fetch("/suggestions")
      .then((resp) => resp.json())
      .then((suggestions) => {
        setSuggestions(suggestions);
      });
  };

  useEffect(() => {
    fetchSuggestions();
  }, []);

  const onRequestRandomPattern = () => {
    setRandomPattern(randomEncoding());
  };

  const onRequestRandomPatternFromBasis = async (encoding: Encoding) => {
    setRandomPattern(encoding);
  };

  const onCloseDialog = () => {
    setRandomPattern(undefined);
  };

  const onRefresh = () => {
    window.location.reload();
  };

  return (
    <Grid container direction="column" spacing={2}>
      <Grid item>
        <Card>
          <CardHeader
            title={
              <Typography variant="h2">
                Sparse Interpretible Audio Demo{" "}
                <Typography variant="caption">(2024-1-28)</Typography>
              </Typography>
            }
          ></CardHeader>
          <CardContent>
            <Dialog
              open={randomPattern !== undefined}
              onClose={onCloseDialog}
              maxWidth="xl"
            >
              <DialogTitle>Random Pattern</DialogTitle>
              <DialogContent>
                {randomPattern !== undefined && (
                  <Stack>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={onRequestRandomPattern}
                      endIcon={<Refresh />}
                    >
                      Refresh
                    </Button>
                    <SegmentEditor
                      encoding={randomPattern}
                      vectorMin={-1}
                      vectorMax={1}
                      nTimeSteps={N_TIME_STEPS}
                    />
                  </Stack>
                )}
              </DialogContent>
              <DialogActions>
                <Button
                  onClick={onCloseDialog}
                  variant="outlined"
                  color="primary"
                >
                  Done
                </Button>
              </DialogActions>
            </Dialog>
            <Stack spacing={2}>
              <Grid item>
                <Typography variant="body1">
                  <p>
                    The model decomposes short (~1.5s) of audio into two
                    components:
                  </p>
                  <ul>
                    <li>A sparse set of "events"</li>
                    <li>
                      A 16-dimensional context vector, which determines global
                      properties of the segment
                    </li>
                  </ul>
                  <p>
                    You can read about the model in more detail here:
                    <a href="https://blog.cochlea.xyz/machine-learning/2023/11/15/sparse-physical-model.html">
                      blog post
                    </a>
                  </p>
                </Typography>
              </Grid>
              <Grid item container direction="row" spacing={2}>
                <Grid item>
                  <Button
                    onClick={onRequestRandomPattern}
                    variant="outlined"
                    color="primary"
                    endIcon={<Equalizer />}
                  >
                    Random Pattern
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    onClick={onRefresh}
                    variant="outlined"
                    color="primary"
                    endIcon={<Refresh />}
                  >
                    Load New Reconstructions
                  </Button>
                </Grid>
              </Grid>
              <Grid item>{suggestions.length === 0 && <LinearProgress />}</Grid>
              {suggestions.slice(0, nReconstructions).map((suggestion) => (
                <Reconstruction
                  suggestion={suggestion}
                  randomPatternRequested={onRequestRandomPatternFromBasis}
                />
              ))}
            </Stack>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default ModelDemoView;
