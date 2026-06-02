"""Product data models for the e-commerce catalog."""

from pydantic import BaseModel, Field


class ProductSpecifications(BaseModel):
    """Product specifications / attributes."""

    material: str | None = None
    weight: str | None = None
    colors: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    dimensions: str | None = None
    battery_life: str | None = None
    connectivity: str | None = None
    warranty: str | None = None


class ProductInventory(BaseModel):
    """Real-time inventory status."""

    in_stock: bool = True
    quantity: int = 0
    warehouse: str = "US-EAST"


class InventoryStatus(BaseModel):
    """Inventory lookup result for a product variant."""

    product_id: str
    in_stock: bool
    quantity: int = Field(ge=0)
    size: str | None = None
    color: str | None = None
    warehouse: str = "US-EAST"


class ProductRating(BaseModel):
    """Aggregated product ratings."""

    average: float = Field(ge=0.0, le=5.0, default=0.0)
    count: int = 0


class Product(BaseModel):
    """Full product document — mirrors the Cosmos DB schema."""

    id: str
    name: str
    category: str
    subcategory: str
    brand: str
    price: float = Field(ge=0.0)
    original_price: float | None = None
    discount_percent: int | None = Field(default=None, ge=0, le=100)
    currency: str = "USD"
    description: str = ""
    specifications: ProductSpecifications = Field(default_factory=ProductSpecifications)
    inventory: ProductInventory = Field(default_factory=ProductInventory)
    rating: ProductRating = Field(default_factory=ProductRating)
    tags: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    partition_key: str = ""  # Same as category for Cosmos DB

    def model_post_init(self, __context) -> None:
        """Auto-set partition_key from category."""
        if not self.partition_key:
            self.partition_key = self.category


class ProductCard(BaseModel):
    """Lightweight product display model for recommendation responses."""

    product_id: str
    name: str
    brand: str
    price: float
    original_price: float | None = None
    discount_percent: int | None = None
    rating: float
    review_count: int
    in_stock: bool
    image_url: str | None = None
    category: str = ""
    subcategory: str = ""

    @classmethod
    def from_product(cls, product: Product) -> "ProductCard":
        """Create a ProductCard from a full Product document."""
        return cls(
            product_id=product.id,
            name=product.name,
            brand=product.brand,
            price=product.price,
            original_price=product.original_price,
            discount_percent=product.discount_percent,
            rating=product.rating.average,
            review_count=product.rating.count,
            in_stock=product.inventory.in_stock,
            image_url=product.images[0] if product.images else None,
            category=product.category,
            subcategory=product.subcategory,
        )


class ProductQuery(BaseModel):
    """Structured product search query — translated from natural language by the LLM."""

    query_text: str | None = Field(default=None, description="Original natural language query")
    category: str | None = None
    subcategory: str | None = None
    brand: str | None = None
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    tags: list[str] = Field(default_factory=list)
    sort_by: str = Field(
        default="rating",
        description="Sort field: rating, price_asc, price_desc, newest",
    )
    in_stock_only: bool = True
    limit: int = Field(default=5, ge=1, le=20)