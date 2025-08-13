#!/usr/bin/env python3
"""
chart_library.py - Chart drawing library for stock price visualization

Based on trend_submit/Data/chart_library.py
Contains DrawOHLC class for generating candlestick chart images.
"""

import numpy as np
import math
from PIL import Image, ImageDraw


class DrawOHLC(object):
    """
    Class for drawing OHLC (Open, High, Low, Close) stock price charts.
    
    Converts normalized stock price data into visual chart images that can be
    used as input for CNN models.
    """
    
    def __init__(self, df, image_width, image_height, has_volume_bar=False, ma_lags=None):
        """
        Initialize OHLC chart drawer.
        
        Args:
            df (pd.DataFrame): Stock price data with OHLC columns (normalized, first Close = 1.0)
            image_width (int): Chart width in pixels
            image_height (int): Chart height in pixels
            has_volume_bar (bool): Whether to include volume bars at bottom
            ma_lags (list): List of moving average periods to overlay
        """
        # Verify data is properly normalized (first close price should be 1.0)
        if np.around(df.iloc[0]["close"], decimals=3) != 1.000:
            raise ValueError("Close on first day not equal to 1.")
        
        # Store configuration parameters
        self.has_volume_bar = has_volume_bar
        self.vol = df["volume"] if has_volume_bar and "volume" in df.columns else None
        self.ma_lags = ma_lags
        self.ma_name_list = (
            ["ma" + str(ma_lag) for ma_lag in ma_lags] if ma_lags is not None else []
        )

        # Prepare price data
        self.df = df[["open", "high", "low", "close"] + self.ma_name_list].abs()

        # Validate data length and calculate price range
        self.ohlc_len = len(df)
        assert self.ohlc_len in [5, 20, 60]  # Supported time windows
        self.minp = self.df.min().min()  # Minimum price for scaling
        self.maxp = self.df.max().max()  # Maximum price for scaling

        # Set chart dimensions
        self.ohlc_width = image_width
        self.ohlc_height = image_height
        
        # Calculate x-axis positions for each trading day
        bar_width = 3  # 3 pixels per day
        first_center = (bar_width - 1) / 2.0
        self.centers = np.arange(
            first_center,
            first_center + bar_width * self.ohlc_len,
            bar_width,
            dtype=int,
        )

    def __ret_to_yaxis(self, ret):
        """
        Convert price/return value to y-axis pixel coordinate.
        
        Args:
            ret (float): Price or return value to convert
        
        Returns:
            int: Y-coordinate in pixels (0 = bottom, height-1 = top)
        """
        pixels_per_unit = (self.ohlc_height - 1.0) / (self.maxp - self.minp)
        res = np.around((ret - self.minp) * pixels_per_unit)
        return int(res)

    def draw_image(self):
        """
        Generate the complete chart image.
        
        Returns:
            PIL.Image: Generated chart image, or None if invalid data
        """
        # Check for invalid price data
        if self.maxp == self.minp or math.isnan(self.maxp) or math.isnan(self.minp):
            return None
        
        # Verify coordinate transformation is valid
        try:
            assert (
                self.__ret_to_yaxis(self.minp) == 0
                and self.__ret_to_yaxis(self.maxp) == self.ohlc_height - 1
            )
        except (ValueError, AssertionError):
            return None

        # Draw main OHLC chart
        ohlc = self.__draw_ohlc()
        
        # For now, just return OHLC without volume
        # Volume can be added later if needed
        image = ohlc

        # Critical: Flip image vertically (PIL uses top-left origin, charts use bottom-left)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __draw_ohlc(self):
        """
        Draw the main OHLC chart with optional moving average overlays.
        
        Returns:
            PIL.Image: OHLC chart image
        """
        ohlc = Image.new(
            "L", (self.ohlc_width, self.ohlc_height), 0  # Black background
        )
        pixels = ohlc.load()
        
        # Draw moving average lines first (background)
        for ma_name in self.ma_name_list:
            if ma_name in self.df.columns:
                ma = self.df[ma_name]
                draw = ImageDraw.Draw(ohlc)
                # Connect moving average points across days
                for day in range(self.ohlc_len - 1):
                    if np.isnan(ma[day]) or np.isnan(ma[day + 1]):
                        continue
                    
                    # Draw line connecting consecutive MA points
                    draw.line(
                        (
                            self.centers[day],
                            self.__ret_to_yaxis(ma[day]),
                            self.centers[day + 1],
                            self.__ret_to_yaxis(ma[day + 1]),
                        ),
                        width=1,
                        fill=255,  # White
                    )
                
                # Draw the final MA point
                try:
                    if not np.isnan(ma[self.ohlc_len - 1]):
                        pixels[
                            int(self.centers[self.ohlc_len - 1]),
                            self.__ret_to_yaxis(ma[self.ohlc_len - 1]),
                        ] = 255
                except (ValueError, IndexError):
                    pass
                del draw

        # Draw OHLC bars/candlesticks for each day
        for day in range(self.ohlc_len):
            highp_today = self.df["high"].iloc[day]
            lowp_today = self.df["low"].iloc[day]
            closep_today = self.df["close"].iloc[day]
            openp_today = self.df["open"].iloc[day]

            if np.isnan(highp_today) or np.isnan(lowp_today):
                continue
            
            # Calculate bar boundaries
            bar_width = 3
            line_width = 3
            
            left = int(math.ceil(self.centers[day] - int(bar_width / 2)))
            right = int(math.floor(self.centers[day] + int(bar_width / 2)))

            line_left = int(math.ceil(self.centers[day] - int(line_width / 2)))
            line_right = int(math.floor(self.centers[day] + int(line_width / 2)))

            # Convert prices to y-coordinates
            line_bottom = self.__ret_to_yaxis(lowp_today)
            line_up = self.__ret_to_yaxis(highp_today)

            # Draw high-low line (thick vertical line from low to high)
            for i in range(line_left, line_right + 1):
                for j in range(line_bottom, line_up + 1):
                    if 0 <= i < self.ohlc_width and 0 <= j < self.ohlc_height:
                        pixels[i, j] = 255

            # Draw opening price (left horizontal line)
            if not np.isnan(openp_today):
                open_line = self.__ret_to_yaxis(openp_today)
                for i in range(left, int(self.centers[day]) + 1):
                    if 0 <= i < self.ohlc_width and 0 <= open_line < self.ohlc_height:
                        pixels[i, open_line] = 255

            # Draw closing price (right horizontal line)
            if not np.isnan(closep_today):
                close_line = self.__ret_to_yaxis(closep_today)
                for i in range(int(self.centers[day]) + 1, right + 1):
                    if 0 <= i < self.ohlc_width and 0 <= close_line < self.ohlc_height:
                        pixels[i, close_line] = 255

        return ohlc