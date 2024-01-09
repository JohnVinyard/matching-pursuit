import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Slider,
  Typography,
} from "@mui/material";

interface VectorProps {
  vector: number[];
  min: number;
  max: number;
  onVectorChange: (vector: number[]) => void;
  readonly?: boolean;
}

const Vector: React.FC<VectorProps> = ({
  vector,
  min,
  max,
  onVectorChange,
  readonly = false,
}) => {
  const sliderChange = (newValue: number, index: number) => {
    const newVector = [...vector];
    newVector[index] = newValue;
    onVectorChange(newVector);
  };

  return (
    <Card>
      <CardHeader
        title={<Typography variant="h6">Context Vector</Typography>}
      ></CardHeader>
      <CardContent>
        <Grid container direction="row" spacing={1}>
          {vector.map((v, i) => (
            <Grid item>
              <Slider
                size="medium"
                orientation="vertical"
                style={{ height: "100px", width: "20px" }}
                disabled={readonly}
                min={min}
                max={max}
                value={v}
                onChangeCommitted={(event, value) => sliderChange(v, i)}
              />
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default Vector;
