import {
  SPRITE_SIZE,
  BOT_COLLISION_RADIUS,
  WALKABLE_AREAS,
} from '../utils/properties';

export class SpriteBot {
  public index: number;
  public name: string;
  public color: string;
  private messageInterval: NodeJS.Timeout | null = null;
  private spriteImage: HTMLImageElement | null = null;
  private imageLoaded = false;
  private spriteSheetPath: string;

  // Sprite sheet configuration for your layout
  private readonly SPRITE_FRAME_WIDTH = 64;
  private readonly SPRITE_FRAME_HEIGHT = 64;
  private readonly FRAMES_PER_DIRECTION = 9;

  // Direction mapping to sprite sheet rows (up, left, down, right)
  private readonly DIRECTION_ROWS = {
    up: 0,
    left: 1,
    down: 2,
    right: 3,
  };

  constructor(
    index: number,
    name: string,
    private onMessage: (index: number, message: string) => void,
    spriteSheetPath?: string
  ) {
    this.index = index;
    this.name = name;

    // Assign unique colors for name tags and UI elements
    const colors = ['#F54DA8', '#832121', '#E21414', '#FFE050', '#8138FEFF'];
    this.color = colors[index % colors.length];

    this.spriteSheetPath = spriteSheetPath || this.getSpriteSheetPath(name);

    this.loadSpriteSheet();
  }

  private getSpriteSheetPath(name: string): string {
    const spriteMapping: { [key: string]: string } = {
      agent_Alice: '/sprites/alice.png',
      agent_Bob: '/sprite/bob.png',
      agent_Charlie: '/sprite/charlie.png',
      agent_Dom: '/sprite/dom.png',
      agent_Elise: '/sprite/elise.png ',
    };

    // Return specific sprite for the character, or fallback to index-based naming
    return spriteMapping[name] || `/globe.svg`;
  }

  private async loadSpriteSheet() {
    try {
      this.spriteImage = new Image();
      this.spriteImage.crossOrigin = 'anonymous';

      this.spriteImage.onload = () => {
        this.imageLoaded = true;
        console.log(
          `Sprite sheet loaded for ${this.name}: ${this.spriteSheetPath}`
        );
      };

      this.spriteImage.onerror = (error) => {
        console.error(
          `Failed to load sprite sheet for ${this.name} (${this.spriteSheetPath}):`,
          error
        );
        this.imageLoaded = false;
        // Try fallback sprite
        this.loadFallbackSprite();
      };

      this.spriteImage.src = this.spriteSheetPath;
    } catch (error) {
      console.error(`Error loading sprite sheet for ${this.name}:`, error);
      this.imageLoaded = false;
    }
  }

  private loadFallbackSprite() {
    if (this.spriteSheetPath !== '/sprites/default.png') {
      console.log(`Trying fallback sprite for ${this.name}`);
      this.spriteSheetPath = '/sprites/default.png';
      this.loadSpriteSheet();
    }
  }

  // Calculate sprite frame based on direction and animation (left to right)
  private getSpriteFrame(
    direction: string,
    animationFrame: number,
    isMoving: boolean
  ): { row: number; col: number } {
    const row =
      this.DIRECTION_ROWS[direction as keyof typeof this.DIRECTION_ROWS] || 0;

    if (!isMoving) {
      return { row, col: 0 }; // Standing frame
    }

    // Walking animation: left to right, 4 frames
    const walkCycle =
      Math.floor(animationFrame / 8) % this.FRAMES_PER_DIRECTION;
    return { row, col: walkCycle };
  }

  // Check if position is in a walkable area
  public static isInWalkableArea(x: number, y: number): boolean {
    return WALKABLE_AREAS.some(
      (area) =>
        x >= area.x &&
        x <= area.x + area.width &&
        y >= area.y &&
        y <= area.y + area.height
    );
  }

  // Check collision with other bots
  public static checkBotCollision(
    x: number,
    y: number,
    botStates: any[],
    excludeIndex: number
  ): boolean {
    for (let i = 0; i < botStates.length; i++) {
      if (i === excludeIndex) continue;

      const otherBot = botStates[i];
      if (!otherBot) continue;

      const distance = Math.sqrt(
        Math.pow(x - otherBot.x, 2) + Math.pow(y - otherBot.y, 2)
      );

      if (distance < BOT_COLLISION_RADIUS * 2) {
        return true; // Collision detected
      }
    }
    return false;
  }

  private drawSprite(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    direction: string,
    animationFrame: number,
    isMoving: boolean,
    scaleFactor: number,
    isEliminated = false
  ) {
    if (!this.imageLoaded || !this.spriteImage) {
      this.drawFallbackSprite(ctx, x, y, scaleFactor, isEliminated);
      return;
    }

    const { row, col } = this.getSpriteFrame(
      direction,
      animationFrame,
      isMoving
    );

    const sourceX = col * this.SPRITE_FRAME_WIDTH;
    const sourceY = row * this.SPRITE_FRAME_HEIGHT;

    const destWidth = SPRITE_SIZE * scaleFactor;
    const destHeight = SPRITE_SIZE * scaleFactor;
    const destX = x - destWidth / 2;
    const destY = y - destHeight / 2;

    try {
      if (isEliminated) {
        ctx.globalAlpha = 0.3;
      }

      ctx.drawImage(
        this.spriteImage,
        sourceX,
        sourceY,
        this.SPRITE_FRAME_WIDTH,
        this.SPRITE_FRAME_HEIGHT,
        destX,
        destY,
        destWidth,
        destHeight
      );

      ctx.globalAlpha = 1.0;
    } catch (error) {
      console.error(`Error drawing sprite for ${this.name}:`, error);
      this.drawFallbackSprite(ctx, x, y, scaleFactor, isEliminated);
    }
  }

  private drawFallbackSprite(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    scaleFactor: number,
    isEliminated = false
  ) {
    const size = SPRITE_SIZE * scaleFactor;

    if (isEliminated) {
      ctx.globalAlpha = 0.3;
    }

    ctx.fillStyle = this.color;
    ctx.fillRect(x - size / 2, y - size / 2, size, size * 0.8);

    // Simple face
    ctx.fillStyle = '#000';
    const eyeSize = 3 * scaleFactor;
    ctx.fillRect(x - 8 * scaleFactor, y - 8 * scaleFactor, eyeSize, eyeSize);
    ctx.fillRect(x + 5 * scaleFactor, y - 8 * scaleFactor, eyeSize, eyeSize);
    ctx.fillRect(
      x - 6 * scaleFactor,
      y - 2 * scaleFactor,
      12 * scaleFactor,
      2 * scaleFactor
    );
    ctx.globalAlpha = 1.0;
  }

  public draw(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    direction: string,
    animationFrame: number,
    isMoving: boolean,
    scaleFactor: number,
    message?: string | null,
    isEliminated = false
  ) {
    const scaledX = x * scaleFactor;
    const scaledY = y * scaleFactor;
    const scaledSize = SPRITE_SIZE * scaleFactor;

    // Draw character shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.ellipse(
      scaledX,
      scaledY + scaledSize * 0.4,
      scaledSize * 0.3,
      scaledSize * 0.15,
      0,
      0,
      2 * Math.PI
    );
    ctx.fill();

    // Draw the sprite
    this.drawSprite(
      ctx,
      scaledX,
      scaledY,
      direction,
      animationFrame,
      isMoving,
      scaleFactor,
      isEliminated
    );

    // Draw collision radius (debug - can be removed)
    if (process.env.NODE_ENV === 'development') {
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(
        scaledX,
        scaledY,
        BOT_COLLISION_RADIUS * scaleFactor,
        0,
        2 * Math.PI
      );
      ctx.stroke();
    }

    // Draw message bubble if there's a message
    if (message && !isEliminated) {
      this.drawMessageBubble(
        ctx,
        scaledX,
        scaledY,
        scaledSize,
        '...',
        scaleFactor
      );
    }

    // Draw name tag
    ctx.fillStyle = 'rgba(0, 0, 0, 1)';
    ctx.font = `bold ${14 * scaleFactor}px Arial`;
    ctx.textAlign = 'center';
    const nameY = scaledY - scaledSize / 2 - 6 * scaleFactor;

    const nameWidth = ctx.measureText(this.name).width;
    ctx.fillRect(
      scaledX - nameWidth / 2 - 3 * scaleFactor,
      nameY - 10 * scaleFactor,
      nameWidth + 7 * scaleFactor,
      14 * scaleFactor
    );

    ctx.fillStyle = '#FFF';
    ctx.fillText(this.name, scaledX, nameY);

    if (!this.imageLoaded) {
      ctx.fillStyle = 'rgba(255, 255, 0, 0.7)';
      ctx.font = `${7 * scaleFactor}px Arial`;
      ctx.fillText(
        'Loading...',
        scaledX,
        scaledY + scaledSize / 2 + 12 * scaleFactor
      );
    }
  }

  private drawMessageBubble(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number,
    message: string,
    scaleFactor: number
  ) {
    const maxWidth = 180 * scaleFactor;
    const padding = 10 * scaleFactor;
    const lineHeight = 14 * scaleFactor;

    ctx.font = `${20 * scaleFactor}px Arial`;
    const words = message.split(' ');
    const lines: string[] = [];
    let currentLine = '';

    for (const word of words) {
      const testLine = currentLine + (currentLine ? ' ' : '') + word;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth - padding * 2 && currentLine) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine) lines.push(currentLine);

    const bubbleWidth = Math.min(
      maxWidth,
      Math.max(...lines.map((line) => ctx.measureText(line).width)) +
        padding * 2
    );
    const bubbleHeight = lines.length * lineHeight + padding * 2;
    const bubbleX = x - bubbleWidth / 2;
    const bubbleY = y - size / 2 - bubbleHeight - 15 * scaleFactor;

    // Draw bubble
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.beginPath();
    ctx.roundRect(bubbleX, bubbleY, bubbleWidth, bubbleHeight, 6 * scaleFactor);
    ctx.fill();

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1.5 * scaleFactor;
    ctx.stroke();

    // Draw pointer
    ctx.beginPath();
    ctx.moveTo(x, y - size / 2 - 8 * scaleFactor);
    ctx.lineTo(x - 6 * scaleFactor, bubbleY + bubbleHeight);
    ctx.lineTo(x + 6 * scaleFactor, bubbleY + bubbleHeight);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.fill();
    ctx.stroke();

    // Draw text
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    lines.forEach((line, index) => {
      ctx.fillText(
        line,
        bubbleX + bubbleWidth / 2,
        bubbleY + padding + (index + 1) * lineHeight - 3 * scaleFactor
      );
    });
  }

  public destroy() {
    if (this.messageInterval) {
      clearTimeout(this.messageInterval);
      this.messageInterval = null;
    }
  }

  public getSpriteSheetInfo(): { path: string; loaded: boolean } {
    return {
      path: this.spriteSheetPath,
      loaded: this.imageLoaded,
    };
  }
}
