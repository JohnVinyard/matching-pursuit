import { useEffect, useState } from "react";
import { Suggestion } from "../models/Suggestion";
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  LinearProgress,
  Stack,
  Typography,
} from "@mui/material";
import Reconstruction from "./Reconstruction";

interface ModelDemoViewProps {
  nReconstructions: number;
}

const ModelDemoView: React.FC<ModelDemoViewProps> = ({}) => {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);

  useEffect(() => {
    fetch("/suggestions")
      .then((resp) => resp.json())
      .then((suggestions) => {
        setSuggestions(suggestions);
        console.log(suggestions);
      });
  }, []);

  return (
    <Grid container direction="column" spacing={2}>
      <Grid item>
        <Card>
          <CardHeader
            title={
              <Typography variant="h2">
                Sparse Interpretible Audio Demo
              </Typography>
            }
          ></CardHeader>
          <CardContent>
            <Stack spacing={2}>
              <Grid item>{suggestions.length === 0 && <LinearProgress />}</Grid>
              {suggestions.map((suggestion) => (
                <Reconstruction suggestion={suggestion} />
              ))}
            </Stack>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default ModelDemoView;
